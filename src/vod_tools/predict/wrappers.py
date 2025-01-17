from __future__ import annotations

import typing
from typing import Any, Iterable

from vod_tools import dstruct, pipes

PREDICT_IDX_COL_NAME = "__idx__"

T = typing.TypeVar("T", bound=dict)

D_co = typing.TypeVar("D_co", bound=dict, covariant=True)


class DatasetWithIndices(dstruct.SizedDataset[D_co]):
    """This class is used to add the column `IDX_COL` to the batch."""

    def __init__(self, dataset: dstruct.SizedDataset[D_co]):
        self.dataset = dataset

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> D_co:
        """Returns a row from the dataset."""
        batch = self.dataset[item]
        if PREDICT_IDX_COL_NAME in batch:
            raise ValueError(
                f"Column {PREDICT_IDX_COL_NAME} already exists in batch (keys={batch.keys()}. "
                f"Cannot safely add the row index."
            )

        batch[PREDICT_IDX_COL_NAME] = item
        return batch

    def __iter__(self) -> Iterable[dict]:
        """Returns an iterator over the rows of the dataset."""
        for i in range(len(self)):
            yield self[i]


def _safely_fetch_key(row: dict) -> int:
    try:
        return row.pop(PREDICT_IDX_COL_NAME)
    except KeyError as exc:
        raise ValueError(
            f"Column {PREDICT_IDX_COL_NAME} not found in batch. "
            f"Make sure to wrap your dataset with `DatasetWithIndices`."
        ) from exc


def _collate_with_indices(examples: Iterable[dict[str, Any]], *, collate_fn: pipes.Collate, **kwargs: Any) -> dict:
    ids = [_safely_fetch_key(row) for row in examples]
    batch = collate_fn(examples, **kwargs)
    batch[PREDICT_IDX_COL_NAME] = ids
    return batch


class CollateWithIndices(pipes.Collate):
    """Wraps a `Collate` to add the column `IDX_COL` to the batch."""

    def __init__(self, collate_fn: pipes.Collate):  # type: ignore
        self.collate_fn = collate_fn

    def __call__(self, examples: Iterable[dict[str, Any]], **kwargs: Any) -> dict:
        """Collate the rows along with the row indixes (`IDX_COL`)."""
        return _collate_with_indices(examples, collate_fn=self.collate_fn, **kwargs)
