from __future__ import annotations

import dataclasses
import functools
import pathlib
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import datasets
import faiss
import loguru
import numpy as np
import omegaconf
import pydantic
import rich
import rich.status
import torch
import transformers
from lightning import pytorch as pl
from omegaconf import DictConfig
from rich import terminal_theme
from rich.progress import track

from raffle_ds_research.cli import utils as cli_utils
from raffle_ds_research.core.builders import FrankBuilder
from raffle_ds_research.core.builders.retrieval_collate import SearchClientConfig
from raffle_ds_research.core.ml_models import Ranker
from raffle_ds_research.core.ml_models.monitor import Monitor
from raffle_ds_research.tools import index_tools, pipes, predict_tools
from raffle_ds_research.tools.index_tools import faiss_tools
from raffle_ds_research.tools.pipes.utils.misc import keep_only_columns
from raffle_ds_research.tools.utils import loader_config
from raffle_ds_research.utils.pretty import print_metric_groups


@dataclasses.dataclass
class Vectors:
    dataset: dict[str, predict_tools.TensorStoreFactory]
    sections: Optional[predict_tools.TensorStoreFactory] = None


@dataclasses.dataclass
class TrainWithIndexConfigs:
    train_loader: loader_config.DataLoaderConfig
    eval_loader: loader_config.DataLoaderConfig
    predict_loader: loader_config.DataLoaderConfig
    train_collate: cli_utils.DefaultCollateConfig
    eval_collate: cli_utils.DefaultCollateConfig
    faiss: cli_utils.DefaultFaissConfig
    bm25: cli_utils.DefaultBm25Config
    update_freq: Optional[Union[int, list[int]]]

    @pydantic.validator("update_freq", pre=True)
    def _validate_update_freq(cls, v):
        if isinstance(v, omegaconf.ListConfig):
            v = omegaconf.OmegaConf.to_container(v)
        return v

    @classmethod
    def parse(cls, config: DictConfig) -> "TrainWithIndexConfigs":
        # get the dataloader configs
        train_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.train)
        eval_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.eval)
        predict_loader_config = loader_config.DataLoaderConfig(**config.loader_configs.predict)

        # get te collate configs
        train_collate_config = cli_utils.DefaultCollateConfig(**config.collate_configs.train)
        eval_collate_config = cli_utils.DefaultCollateConfig(**config.collate_configs.eval)

        # set the index configs
        faiss_config = cli_utils.DefaultFaissConfig(**config.faiss_config)
        bm25_config = cli_utils.DefaultBm25Config(**config.bm25_config)

        return cls(
            update_freq=config.update_freq,
            train_loader=train_loader_config,
            eval_loader=eval_loader_config,
            predict_loader=predict_loader_config,
            train_collate=train_collate_config,
            eval_collate=eval_collate_config,
            faiss=faiss_config,
            bm25=bm25_config,
        )


def train_with_index_updates(
    ranker: Ranker,
    monitor: Monitor,
    trainer: pl.Trainer,
    builder: FrankBuilder,
    config: TrainWithIndexConfigs | DictConfig,
) -> Ranker:
    """
    Train a ranker while periodically updating the faiss index.
    TODO: refactor. This needs to be simplified, especially how to input configs.
    """
    if isinstance(config, DictConfig):
        config = TrainWithIndexConfigs.parse(config)

    total_number_of_steps = trainer.max_steps
    update_steps = _infer_update_steps(total_number_of_steps, config.update_freq)
    loguru.logger.info(f"Index will be updated at steps: {_pretty_steps(update_steps)}")
    if len(update_steps) == 0:
        raise ValueError("No index update steps were defined.")

    stop_callback = PeriodicStoppingCallback(stop_at=-1)
    trainer.callbacks.append(stop_callback)  # type: ignore

    for period_idx, (start_step, end_step) in enumerate(zip(update_steps[:-1], update_steps[1:])):
        _log_metrics(
            {
                "trainer/period": float(period_idx + 1),
                "trainer/faiss_weight": config.faiss.get_weight(period_idx),
                "trainer/bm25_weight": config.bm25.get_weight(period_idx),
            },
            trainer=trainer,
        )
        loguru.logger.info(
            f"Training period {period_idx + 1}/{len(update_steps) - 1} (step {start_step} -> {end_step})"
        )

        # compute question and section vectors, spin up the faiss index.
        with IndexManager(
            ranker=ranker,
            trainer=trainer,
            builder=builder,
            config=config,
            index_step=period_idx,
        ) as manager:
            train_loader = manager.dataloader("train")
            val_loader = manager.dataloader("validation")

            # run the static validation & log results
            on_first_batch_callback = functools.partial(
                _log_retrieval_batch,
                tokenizer=builder.tokenizer,
                period_idx=period_idx,
                gloabl_step=trainer.global_step,
                max_sections=5,
            )
            static_eval_metrics = _run_static_evaluation(
                loader=track(val_loader, description="Static validation"),
                monitor=monitor,
                on_first_batch=on_first_batch_callback,
            )
            _log_metrics(static_eval_metrics, trainer=trainer, console=True)

            # train for the current period
            trainer.should_stop = False
            stop_callback.stop_at = end_step
            manager.trainer.fit(
                manager.ranker,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

    return ranker


def _run_static_evaluation(
    loader: Iterable[dict[str, Any]],
    monitor: Monitor,
    on_first_batch: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> dict[str, Any]:
    monitor.reset()
    for i, batch in enumerate(loader):
        if i == 0 and on_first_batch is not None:
            on_first_batch(batch)
        monitor.update_from_retrieval_batch(batch, field="section")
    metrics = monitor.compute(prefix="val/")
    monitor.reset()
    return metrics


class IndexManager(object):
    """ "
    build and Spin up the indexes.
    todo: simplify; remove collate_fn initialization form here.
    """

    _dataset: Optional[datasets.DatasetDict] = None
    _tmpdir: Optional[tempfile.TemporaryDirectory] = None
    _faiss_master: Optional[index_tools.FaissMaster] = None
    _bm25_master: Optional[index_tools.Bm25Master] = None
    _collate_fns: Optional[dict[str, pipes.Collate]] = None

    def __init__(
        self,
        *,
        ranker: Ranker,
        trainer: pl.Trainer,
        builder: FrankBuilder,
        config: TrainWithIndexConfigs,
        index_step: int,
    ) -> None:
        self.ranker = ranker
        self.trainer = trainer
        self.builder = builder
        self.config = config
        self.index_step = index_step

    def __enter__(self):
        self._dataset = self.builder()

        # create a temporary working directory
        self._tmpdir = tempfile.TemporaryDirectory(prefix="tmp-training-")
        tmpdir = self._tmpdir.__enter__()
        loguru.logger.info(f"Setting up index at `{tmpdir}`")

        # init the bm25 index
        bm25_weight = self.config.bm25.get_weight(self.index_step)
        if bm25_weight >= 0:
            corpus = self.builder.get_corpus()
            corpus = keep_only_columns(corpus, [self.config.bm25.indexed_key, self.config.bm25.label_key])
            index_name = f"index-{corpus._fingerprint}"
            self._bm25_master = index_tools.Bm25Master(
                texts=(row[self.config.bm25.indexed_key] for row in corpus),
                labels=(row[self.config.bm25.label_key] for row in corpus) if self.config.bm25.use_labels else None,
                input_size=len(corpus),
                index_name=index_name,
                exist_ok=True,
                persistent=True,
            )
            bm25_master = self._bm25_master.__enter__()
        else:
            bm25_master = None

        # init the faiss index
        faiss_weight = self.config.faiss.get_weight(self.index_step)
        if faiss_weight >= 0:
            # compute the vectors for the questions and sections
            vectors = self._compute_vectors(tmpdir)
            vectors = self._compute_vectors(tmpdir)

            # build the faiss index and save to disk
            faiss_index = faiss_tools.build_index(vectors.sections, factory_string=self.config.faiss.factory)
            faiss_path = Path(tmpdir, "index.faiss")
            faiss.write_index(faiss_index, str(faiss_path))

            # spin up the faiss server
            self._faiss_master = index_tools.FaissMaster(faiss_path, self.config.faiss.nprobe)
            faiss_master = self._faiss_master.__enter__()

        else:
            vectors = Vectors(dataset={})
            faiss_master = None

        # Create the `collate_fn` for each split
        self._collate_fns = {}
        for split in self._dataset:
            vecs_for_split = vectors.dataset.get(split, None)
            collate_fn = self._instantiate_collate_for_split(
                split=split,
                vectors=vecs_for_split,
                faiss_master=faiss_master,
                bm25_master=bm25_master,
                faiss_weight=faiss_weight,
                bm25_weight=bm25_weight,
            )
            self._collate_fns[split] = collate_fn

        return self

    def _compute_vectors(self, tmpdir: pathlib.Path) -> Vectors:
        dataset_vectors = _compute_dataset_vectors(
            dataset=self._dataset,
            trainer=self.trainer,
            tokenizer=self.builder.tokenizer,
            model=self.ranker,
            cache_dir=tmpdir,
            field="question",
            loader_config=self.config.predict_loader,
        )
        sections_vectors = _compute_dataset_vectors(
            dataset=self.builder.get_corpus(),
            trainer=self.trainer,
            tokenizer=self.builder.tokenizer,
            model=self.ranker,
            cache_dir=tmpdir,
            field="section",
            loader_config=self.config.predict_loader,
        )
        return Vectors(dataset=dataset_vectors, sections=sections_vectors)

    def _instantiate_collate_for_split(
        self,
        split: str,
        vectors: predict_tools.TensorStoreFactory,
        faiss_master: Optional[index_tools.FaissMaster],
        bm25_master: Optional[index_tools.Bm25Master],
        faiss_weight: float,
        bm25_weight: float,
    ) -> pipes.Collate:
        base_collate_cfg = {
            "train": self.config.train_collate,
        }.get(split, self.config.eval_collate)

        clients = []
        if faiss_weight >= 0:
            clients.append(
                SearchClientConfig(
                    name="faiss",
                    client=faiss_master.get_client(),
                    weight=faiss_weight,
                    label_key=self.config.faiss.label_key,
                )
            )
        if bm25_weight >= 0:
            clients.append(
                SearchClientConfig(
                    name="bm25",
                    client=bm25_master.get_client(),
                    weight=bm25_weight,
                    label_key=self.config.bm25.label_key,
                )
            )

        full_collate_config = self.builder.collate_config(
            question_vectors=vectors,
            clients=clients,
            **base_collate_cfg.dict(),
        )
        collate_fn = self.builder.get_collate_fn(config=full_collate_config)
        return collate_fn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataset = None
        self._collate_fns = None
        if self._faiss_master is not None:
            self._faiss_master.__exit__(exc_type, exc_val, exc_tb)
            self._faiss_master = None
        self._tmpdir.__exit__(exc_type, exc_val, exc_tb)
        self._tmpdir = None

    def dataloader(self, split: str) -> torch.utils.data.DataLoader:
        loader_config = {
            "train": self.config.train_loader,
        }.get(split, self.config.eval_loader)

        dset = self._dataset[split]
        collate_fn = self._collate_fns[split]
        return torch.utils.data.DataLoader(
            dset,
            collate_fn=collate_fn,
            **loader_config.dict(),
        )


def _compute_dataset_vectors(
    *,
    dataset: datasets.Dataset | datasets.DatasetDict,
    model: torch.nn.Module,
    trainer: pl.Trainer,
    cache_dir: Path,
    tokenizer: transformers.PreTrainedTokenizer,
    field: str,
    max_length: int = 512,  # todo: don't hardcode this
    loader_config: loader_config.DataLoaderConfig,
) -> predict_tools.TensorStoreFactory | dict[str, predict_tools.TensorStoreFactory]:
    output_key = {"question": "hq", "section": "hd"}[field]
    collate_fn = functools.partial(
        pipes.torch_tokenize_collate,
        tokenizer=tokenizer,
        prefix_key=f"{field}.",
        max_length=max_length,
        truncation=True,
    )
    return predict_tools.predict(
        dataset,
        trainer=trainer,
        cache_dir=cache_dir,
        model=model,
        model_output_key=output_key,
        collate_fn=collate_fn,
        loader_kwargs=loader_config,
    )


def _infer_update_steps(total_number_of_steps: int, update_freq: int | list[int]) -> list[int]:
    if isinstance(update_freq, int):
        steps = [int(x) for x in np.arange(0, total_number_of_steps, update_freq)]
    elif isinstance(update_freq, list):
        if update_freq[0] != 0:
            update_freq = [0] + update_freq
        if update_freq[-1] == total_number_of_steps:
            update_freq = update_freq[:-1]
        steps = update_freq
    else:
        raise TypeError(f"Invalid type for `update_freq`: {type(update_freq)}")

    return steps + [total_number_of_steps]


def _pretty_steps(steps: list[int]) -> str:
    steps = steps[:-1]
    if len(steps) > 6:
        return f"[{steps[0]}, {steps[1]}, {steps[2]}, {steps[3]}, {steps[4]} ... {steps[-1]}]"
    else:
        return str(steps)


def _log_metrics(metrics: dict[str, Any], trainer: pl.Trainer, console: bool = False):
    for logger in trainer.loggers:
        logger.log_metrics(metrics, step=trainer.global_step)

    if console:
        print_metric_groups(metrics)


def _log_retrieval_batch(
    batch: dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
    period_idx: int,
    gloabl_step: int,
    max_sections: int = 10,
):
    try:
        console = rich.console.Console(record=True)
        pipes.pprint_supervised_retrieval_batch(
            batch,
            header="Evaluation batch",
            tokenizer=tokenizer,
            console=console,
            skip_special_tokens=True,
            max_sections=max_sections,
        )
        html_path = str(Path(f"batch-period{period_idx + 1}.html"))
        console.save_html(html_path, theme=terminal_theme.MONOKAI)

        import wandb

        wandb.log({"trainer/eval-batch": wandb.Html(open(html_path))}, step=gloabl_step)
    except Exception as e:
        loguru.logger.debug(f"Could not log batch to wandb: {e}")


class PeriodicStoppingCallback(pl.callbacks.Callback):
    def __init__(self, stop_at: int):
        super().__init__()
        self.stop_at = stop_at

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step >= self.stop_at:
            trainer.should_stop = True