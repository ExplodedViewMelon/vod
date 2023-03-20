from __future__ import annotations

import collections
import dataclasses
from functools import partial
from typing import Any, Callable, Optional

import datasets
import numpy as np
import pydantic
import torch
import transformers
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel

from raffle_ds_research.core.builders.utils import numpy_gumbel_like, numpy_log_softmax
from raffle_ds_research.tools import c_tools, dataset_builder, index_tools, pipes
from raffle_ds_research.tools.pipes.utils.misc import pack_examples
from raffle_ds_research.tools.raffle_datasets import frank

QUESTION_TEMPLATE = "Question: {{ question }}"
SECTION_TEMPLATE = "{% if title %}Title: {{ title }}. Document: {% endif %}{{ content }}"

DEFAULT_TEMPLATES = {
    "question": QUESTION_TEMPLATE,
    "section": SECTION_TEMPLATE,
}


class FrankRowModel(BaseModel):
    question: str
    answer_id: int
    section_id: Optional[int]
    kb_id: int


def _to_tensor(x: Any, dtype: torch.dtype, replace: Optional[dict] = None) -> torch.Tensor:
    if replace is not None:
        if isinstance(x, list):
            x = [replace.get(i, i) for i in x]
        else:
            raise TypeError(f"Cannot use `replace` with type {type(x)}")

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype=dtype)
    elif isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype)
    else:
        x = torch.tensor(x, dtype=dtype)

    return x


class FrankLoaderConfig(BaseModel):
    """Handles the variables required to instantiate the `collate_fn` for the `FrankBuilder`."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    split: str = "train"
    n_sections: int = 10
    prefetch_n_sections: int = 100
    max_pos_sections: int = 3
    sample_negatives: bool = False
    question_max_length: int = 512
    section_max_length: int = 512
    question_vectors: Optional[index_tools.VectorType] = None
    _query_vectors: index_tools.VectorHandler = pydantic.PrivateAttr(None)
    faiss_client: Optional[index_tools.FaissClient] = None
    label_keys: collections.OrderedDict[str, str] = {"answer_id": "answer_id", "section_id": "id"}
    in_domain_keys: collections.OrderedDict[str, str] = {"kb_id": "kb_id"}

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.question_vectors is not None:
            self._query_vectors = index_tools.vector_handler(self.question_vectors)

    def __getstate__(self):
        """Drop the open ts.TensorStore object to make the state serializable."""
        state = super().__getstate__().copy()
        state["__private_attribute_values__"].pop("_query_vectors")
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.question_vectors is not None:
            self._query_vectors = index_tools.vector_handler(self.question_vectors)

    def __del__(self):
        """Close the open ts.TensorStore object."""
        if self._query_vectors is not None:
            del self._query_vectors

    @property
    def query_vectors(self) -> index_tools.VectorHandler:
        return self._query_vectors

    @property
    def faiss_not_available(self) -> bool:
        return self.question_vectors is None or self.faiss_client is None

    @pydantic.validator("label_keys", pre=True)
    def _validate_label_keys(cls, v: Any) -> collections.OrderedDict[str, str]:
        if isinstance(v, collections.OrderedDict):
            return v
        elif isinstance(v, dict):
            return collections.OrderedDict(v)
        else:
            raise TypeError(f"Expected dict or OrderedDict, got {type(v)}")


def _wrap_as_retrieval_batch(
    lookup_results: pipes.LookupSearchResults,
    is_defined_rule: Optional[Callable[[pipes.LookupSearchResults], np.ndarray]] = None,
) -> index_tools.RetrievalBatch:
    """Wrap the lookup results as a `RetrievalBatch`."""
    if is_defined_rule is None:
        is_defined = lookup_results.frequencies.sum(axis=-1) > 0
    else:
        is_defined = is_defined_rule(lookup_results)

    # override the scores to -1
    scores = np.where(is_defined, 0.0, -np.inf)
    indices = np.where(is_defined, lookup_results.indices, -1)
    return index_tools.RetrievalBatch(
        indices=indices,
        scores=scores,
    )


def _answer_or_section_rule(results: pipes.LookupSearchResults) -> np.ndarray:
    """For each row, set mark the sections as positive.
    - where there is a match on `id` (section_id).
    - else where there is a match on `answer_id` (answer_id).
    """
    id_col_idx = results.labels.index("id")
    answer_id_col_idx = results.labels.index("answer_id")

    # frequencies: [batch_size, n_sections, n_labels]
    frequencies: np.ndarray = results.frequencies

    # compute the `id_defined` mask
    has_id_match = frequencies[:, :, id_col_idx].any(axis=-1)
    is_defined = np.where(
        has_id_match[:, None],
        frequencies.sum(axis=-1) > 0,
        frequencies[:, :, answer_id_col_idx] > 0,
    )
    return is_defined


def sample_sections(
    *,
    positives: index_tools.RetrievalBatch,
    negatives: index_tools.RetrievalBatch,
    n_sections: int,
    max_pos_sections: int,
    sample_negatives: bool = True,
) -> SampledSections:
    """Sample the positive and negative sections.
    This function uses the Gumbel-Max trick to sample from the corresponding distributions.
    Gumbel-Max: https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    """
    if max_pos_sections is None:
        max_pos_sections = n_sections

    # gather the positive sections
    positive_indices = positives.indices
    positive_scores = positives.scores
    positive_logits = numpy_log_softmax(positive_scores)
    positive_logits += numpy_gumbel_like(positive_logits)

    # Define the positive scores:
    # set the positive scores to NaN by default
    positive_scores = np.where(np.isinf(positive_scores), -np.inf, np.nan)
    # replace the values of the positive scores when available in the pool of negatives (returned by faiss)
    positive_scores = c_tools.copy_by_index(
        a_indices=positive_indices,
        a_values=positive_scores,
        b_indices=negatives.indices,
        b_values=negatives.scores,
    )

    # gather the negative sections
    negative_indices = negatives.indices
    negative_scores = negatives.scores
    negative_logits = numpy_log_softmax(negative_scores)
    if sample_negatives:
        negative_logits += numpy_gumbel_like(negative_logits)

    # concat the positive and negative sections
    concatenated = c_tools.concat_search_results(
        a_indices=positive_indices,
        a_scores=positive_logits,
        a_features=positive_scores,
        b_indices=negative_indices,
        b_scores=negative_logits,
        b_features=negative_scores,
        max_a=max_pos_sections,
        total=n_sections,
    )
    concatenated.labels = np.where(concatenated.labels == 0, 1, 0)
    return SampledSections(
        indices=concatenated.indices,
        scores=concatenated.features,
        labels=concatenated.labels,
    )


class FrankCollate(pipes.Collate):
    def __init__(
        self,
        *,
        corpus: datasets.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        config: FrankLoaderConfig,
        **kwargs: Any,
    ):
        self.corpus = corpus
        self.lookup_index = pipes.LookupIndexPipe(corpus, keys=list(config.label_keys.values()))
        self.in_domain_lookup_index = pipes.LookupIndexPipe(corpus, keys=list(config.in_domain_keys.values()))
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        batch = pack_examples(examples)

        # make the question extras
        questions_extras_keys = ["id", "section_id", "answer_id", "kb_id"]
        questions_extras = {
            f"question.{k}": _to_tensor(batch[k], dtype=torch.long, replace={None: -1}) for k in questions_extras_keys
        }

        # fetch the negative section ids (using faiss, or by fetching sections from the same kb_id)
        if self.config.faiss_not_available:
            neg_lookup_input = {v: batch[k] for k, v in self.config.in_domain_keys.items()}
            negative_samples: index_tools.RetrievalBatch = _wrap_as_retrieval_batch(
                self.in_domain_lookup_index.search(neg_lookup_input)
            )
        else:
            question_ids = batch[dataset_builder.ROW_IDX_COL_NAME]
            query_vectors = self.config.query_vectors[question_ids]
            negative_samples: index_tools.RetrievalBatch = self.config.faiss_client.search(
                query_vectors,
                top_k=self.config.prefetch_n_sections,
            )

        # tokenize the questions
        tokenized_question = pipes.torch_tokenize_pipe(
            batch,
            tokenizer=self.tokenizer,
            field="question",
            max_length=self.config.question_max_length,
            truncation=True,
        )

        # fetch the positive section ids (The `rule` allows keeping sections with match on either `id` or `answer_id`)
        # Todo: fetch the model scores for the positive samples. This can be done using the `faiss_client` or
        #  using the `query_vectors` and by fetching the section vectors.
        pos_lookup_input = {v: batch[k] for k, v in self.config.label_keys.items()}
        positive_samples: index_tools.RetrievalBatch = _wrap_as_retrieval_batch(
            self.lookup_index.search(pos_lookup_input),
            is_defined_rule=_answer_or_section_rule,
        )

        # sample the sections given
        sections: SampledSections = sample_sections(
            negatives=negative_samples,
            positives=positive_samples,
            n_sections=self.config.n_sections,
            max_pos_sections=self.config.max_pos_sections,
            sample_negatives=self.config.sample_negatives,
        )

        # fetch the content of each section
        flat_ids = sections.indices.flatten().tolist()
        flat_sections_content = self.corpus[flat_ids]

        # tokenize the sections and add them to the output
        tokenized_sections = pipes.torch_tokenize_pipe(
            flat_sections_content,
            tokenizer=self.tokenizer,
            field="section",
            max_length=self.config.section_max_length,
            truncation=True,
        )
        tokenized_sections = {k: v.view(*sections.indices.shape, -1) for k, v in tokenized_sections.items()}

        # make the section extras
        section_extras_keys = ["id", "answer_id", "kb_id"]
        sections_extras = {
            f"section.{k}": _to_tensor(flat_sections_content[k], dtype=torch.long) for k in section_extras_keys
        }
        sections_extras = {k: v.view(*sections.indices.shape) for k, v in sections_extras.items()}

        return {
            **questions_extras,
            **tokenized_question,
            **tokenized_sections,
            **sections.to_dict(prefix="section.", as_torch=True),
            **sections_extras,
        }


@dataclasses.dataclass
class SampledSections:
    indices: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def to_dict(self, prefix: str = "", as_torch: bool = False) -> dict[str, np.ndarray | torch.Tensor]:
        output = {
            f"{prefix}idx": self.indices,
            f"{prefix}score": self.scores,
            f"{prefix}label": self.labels,
        }

        if as_torch:
            output = {k: torch.from_numpy(v) for k, v in output.items()}
            output[f"{prefix}label"] = output[f"{prefix}label"].to(torch.bool)

        return output


class FrankBuilder(dataset_builder.HfBuilder):
    _collate_config = FrankLoaderConfig

    def __init__(
        self,
        name: str = "frank",
        subset_name: str = "A",
        language: str = "en",
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        prep_map_kwargs: Optional[dict] = None,
        index_max_top_k: int = 100,
        n_sections: int = 32,
        question_max_length: Optional[int] = 512,
        section_max_length: Optional[int] = 512,
        templates: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            subset_name=subset_name,
            row_model=FrankRowModel,
            batch_model=None,
            hf_load_kwargs=None,
            prep_map_kwargs=prep_map_kwargs,
            **kwargs,
        )

        if templates is None:
            templates = DEFAULT_TEMPLATES

        self.language = language
        self.frank_split = frank.FrankSplitName(self.subset_name)
        self.tokenizer = tokenizer
        self.index_max_top_k = index_max_top_k

        # store the parameters for preprocessing
        self.templates = templates

        # collate
        self.n_sections = n_sections
        self.question_max_length = question_max_length
        self.section_max_length = section_max_length

    def _load_frank_split(self, frank_split: frank.FrankSplitName) -> frank.HfFrankSplit:
        return frank.load_frank(self.language, split=frank_split)

    def _build_sections(self) -> datasets.Dataset:
        frank_split = self._load_frank_split(self.frank_split)
        sections = frank_split.sections
        pipe = self._get_sections_preprocessing()
        sections = sections.map(
            pipe,
            **self.prep_map_kwargs(desc=f"Preprocessing Frank ({self.frank_split}) sections"),
        )
        return sections

    def _build_dset(self, corpus: Optional[datasets.Dataset]) -> datasets.DatasetDict:
        frank_split = self._load_frank_split(self.frank_split)
        return frank_split.qa_splits

    def _get_sections_preprocessing(self) -> pipes.Pipe:
        section_prep = partial(
            pipes.template_pipe,
            template=self.templates["section"],
            input_keys=["title", "content"],
            output_key="section",
        )
        return section_prep

    def get_corpus(self) -> Optional[datasets.Dataset]:
        return self._build_sections()

    def get_collate_fn(self, config: Optional[FrankLoaderConfig] = None):
        if config is None:
            config = FrankLoaderConfig()
        if isinstance(config, (dict, DictConfig)):
            config = FrankLoaderConfig(**config)

        if config.faiss_not_available:
            logger.debug(
                "Disabling `faiss` search in this instance of `collate_fn` "
                "(missing question_vectors` or `faiss_client`)"
            )

        sections = self.get_corpus()
        collate_fn = FrankCollate(
            corpus=sections,
            tokenizer=self.tokenizer,
            config=config,
        )
        return collate_fn
