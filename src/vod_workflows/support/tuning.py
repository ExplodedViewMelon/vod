from __future__ import annotations

import dataclasses
import pathlib
import time
import typing
from typing import Any

import lightning as L
import torch
import transformers
from loguru import logger
from torch import nn
from vod_tools import dstruct
from vod_tools.misc.progress import IterProgressBar
from vod_workflows.utils import helpers

from src import vod_configs, vod_datasets, vod_gradients, vod_search

_DEFAULT_TUNE_LIST = ["bm25"]


def _min_score_no_nan(score: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the minimum score along a dimension, ignoring NaNs."""
    min_scores_along_dim, _ = torch.min(
        torch.where(torch.isnan(score), torch.tensor(float("inf")), score),
        dim=dim,
        keepdim=True,
    )

    return torch.where(torch.isinf(min_scores_along_dim), 0, min_scores_along_dim)


class HybridRanker(nn.Module):
    """A ranker that combines multiple scores."""

    def __init__(self, parameters: dict[str, Any], require_grads: list[str], nan_offset: float = 0.0) -> None:
        super().__init__()
        self.grad_fn = vod_gradients.SupervisedRetrievalGradients()
        self.params = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(v), requires_grad=k in require_grads) for k, v in parameters.items()}
        )
        self.nan_offset = nan_offset

    def pydict(self) -> dict[str, Any]:
        """Return a python dictionary of the parameters."""
        return {k: v.item() for k, v in self.params.items()}

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute the hybrid score."""
        hybrid_score = None
        for key, weight in self.params.items():
            batch_key = f"section.{key}"
            if batch_key not in batch:
                continue

            # Fetch the score
            score = batch[batch_key]
            min_scores_along_dim = _min_score_no_nan(score, dim=1)
            score = torch.where(torch.isnan(score), min_scores_along_dim + self.nan_offset, score)

            # Add the score to the hybrid score
            hybrid_score = weight * score if hybrid_score is None else hybrid_score + weight * score

        if hybrid_score is None:
            raise ValueError("No hybrid score was computed")

        # Compute the gradients
        hybrid_logprobs = torch.log_softmax(hybrid_score, dim=-1)
        return self.grad_fn(batch, retriever_logprobs=hybrid_logprobs)


K = typing.TypeVar("K")


def get_total_duration(duration: str) -> float:
    """Return the total duration in seconds."""
    max_duration = vod_configs.RE_DURATION_IN_SECONDS.match(duration)
    if max_duration is not None:
        return float(max_duration.group(1))

    max_duration = vod_configs.RE_DURATION_IN_MINUTES.match(duration)
    if max_duration is not None:
        return float(max_duration.group(1)) * 60

    raise ValueError(f"Unknown duration format: {duration}")


def should_stop(
    step: int,
    elapsed_time: float,
    total_steps: None | int,
    total_time: None | float,
) -> bool:
    """Return whether the tuning should stop."""
    return (total_steps is not None and step >= total_steps) or (total_time is not None and elapsed_time >= total_time)


class NanLossError(ValueError):
    """Raised when a loss is NaN."""


def tune_parameters(
    parameters: dict[str, float],
    tune: None | list[str] = None,
    *,
    fabric: L.Fabric,
    factories: dict[K, vod_datasets.DatasetFactory],
    vectors: dict[K, helpers.PrecomputedDsetVectors],
    search_config: vod_configs.SearchConfig,
    collate_config: vod_configs.RetrievalCollateConfig,
    dataloader_config: vod_configs.DataLoaderConfig,
    cache_dir: pathlib.Path,
    serve_on_gpu: bool = True,
    tuning_steps: int | str = "3min",
    learning_rate: float = 1e-3,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
) -> dict[str, float]:
    """Run benchmarks on a retrieval task."""
    tune = tune or _DEFAULT_TUNE_LIST
    if not all(t in parameters for t in tune):
        raise ValueError(f"tune list `{tune}` contains unknown parameters")

    if fabric.is_global_zero:
        # Define the model
        model = HybridRanker(parameters=parameters, require_grads=tune, nan_offset=-100)

        # make task
        task = _make_tuning_task(
            factories=factories,
            vectors=vectors,
        )

        with vod_search.build_multi_search_engine(
            sections=task.sections.data,
            vectors=task.sections.vectors,
            config=search_config,
            cache_dir=cache_dir,
            faiss_enabled=True,
            bm25_enabled=True,
            serve_on_gpu=serve_on_gpu,
        ) as master:
            search_client = master.get_client()

            # instantiate the dataloader
            dataloader = helpers.instantiate_retrieval_dataloader(
                questions=task.questions,
                sections=task.sections,
                tokenizer=tokenizer,
                search_client=search_client,
                collate_config=collate_config,
                dataloader_config=dataloader_config,
                parameters=parameters,
            )

            output = {}
            step = 0
            try:
                # Optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                # Define the total duration
                if isinstance(tuning_steps, str):
                    total_duration = get_total_duration(tuning_steps)
                    total_steps = None
                else:
                    total_duration = None
                    total_steps = tuning_steps

                with IterProgressBar() as pbar:
                    ptask = pbar.add_task(
                        "Tuning parameters",
                        total=total_steps or total_duration,
                        info=_info_bar(output=output, model=model),
                    )
                    t_0 = time.perf_counter()
                    tick = time.perf_counter()
                    while not should_stop(
                        step,
                        time.perf_counter() - t_0,
                        total_steps=total_steps,
                        total_time=total_duration,
                    ):
                        for batch in dataloader:
                            output = model(batch)
                            loss = output["loss"]
                            if torch.isnan(loss):
                                raise NanLossError("NaN loss")
                            loss.backward()

                            # Update the parameters
                            optimizer.step()
                            optimizer.zero_grad()

                            # Update the progress bar
                            step += 1
                            pbar.update(
                                ptask,
                                advance=1 if total_steps else time.perf_counter() - tick,
                                info=_info_bar(output=output, model=model),
                            )
                            tick = time.perf_counter()

            except KeyboardInterrupt:
                logger.warning(f"Parameter tuning interrupted at step {step} (KeyboardInterrupt).")

            except NanLossError:
                logger.error(f"Parameter tuning aborted at step {step} (NaN loss).")

        # update the parameters
        parameters = model.pydict()

    # broadcast the metrics to all the workers
    return fabric.broadcast(parameters)


def _info_bar(output: dict[str, Any], model: HybridRanker) -> str:
    """Return the info bar."""
    base = ""
    if "loss" in output:
        base += f" loss={output['loss'].item():.3f}"

    for param, value in model.pydict().items():
        base += f" {param}={value:.3f}"

    return base


@dataclasses.dataclass(frozen=True)
class TuningTask:
    """Holds the train and validation datasets."""

    questions: helpers.DsetWithVectors
    sections: helpers.DsetWithVectors


def _make_tuning_task(
    factories: dict[K, vod_datasets.DatasetFactory],
    vectors: dict[K, helpers.PrecomputedDsetVectors],
) -> TuningTask:
    """Create the `RetrievalTask` from the training and validation factories."""

    def _vec(key: K, field: typing.Literal["question", "section"]) -> dstruct.TensorStoreFactory:
        """Safely fetch the relevant `vector` from the `PrecomputedDsetVectors` structure."""
        x = vectors[key]
        if field == "question":
            return x.questions
        if field == "section":
            return x.sections
        raise ValueError(f"Unknown field: {field}")

    return TuningTask(
        questions=helpers.concatenate_datasets(
            [
                helpers.DsetWithVectors.cast(data=factory.get_qa_split(), vectors=_vec(key, "question"))
                for key, factory in factories.items()
            ]
        ),
        sections=helpers.concatenate_datasets(
            [
                helpers.DsetWithVectors.cast(data=factory.get_sections(), vectors=_vec(key, "section"))
                for key, factory in factories.items()
            ]
        ),
    )
