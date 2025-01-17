from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional, TypeVar

import datasets
import numpy as np

T = TypeVar("T")


def iter_examples(batch: dict[str, list], keys: Iterable[str] = None) -> Iterable[dict]:
    """Iterate over the examples contained in a batch."""
    keys = list(batch.keys()) if keys is None else list(keys)
    subset = {key: batch[key] for key in keys}
    master_key, *other_keys = subset
    for i in range(len(batch[master_key])):
        example = {key: batch[key][i] for key in keys}
        yield example


def pack_examples(examples: Iterable[dict[T, Any]], keys: Optional[list[T]] = None) -> dict[T, list[Any]]:
    """Pack a list of examples into a batch."""
    output = defaultdict(list)
    for example in examples:
        if keys is None:
            keys = set(example.keys())
        elif not set(keys).issubset(example.keys()):
            raise ValueError(f"Expected keys {set(keys)}, got {set(example.keys())}")
        for key in keys:
            output[key].append(example[key])
    return dict(output)


_FILL_VALUE_UNSET = object()


def pad_list(
    x: list[T],
    length: int,
    fill_value: T = _FILL_VALUE_UNSET,
    fill_values: Optional[list[T]] = None,
) -> list[T]:
    """Pad a list to a given length."""
    if fill_values is not None and fill_value is not None:
        raise ValueError("`fill_value` and `fill_values` cannot be both set")
    n_missing = length - len(x)
    if n_missing <= 0:
        return x[:length]

    if fill_values is None:
        if fill_value is _FILL_VALUE_UNSET:
            raise ValueError("Must set either `fill_value` or `fill_values`")
        return x + [fill_value] * n_missing

    if isinstance(fill_values, set):
        fill_values = list(fill_values)
    samples = np.random.choice(fill_values, n_missing, replace=n_missing > len(fill_values))
    samples = samples.tolist()
    return x + samples


def keep_only_columns(dataset: datasets.Dataset, columns: Iterable[str], strict: bool = True) -> datasets.Dataset:
    """Keep only the specified columns in a `datasets.Dataset`."""
    columns = set(columns)
    if strict and not columns.issubset(dataset.column_names):
        raise ValueError(
            f"Columns {columns - set(dataset.column_names)} not in dataset and are required with argument `strict=True`"
        )
    cols_to_remove = set(dataset.column_names) - columns
    cols_to_remove = sorted(cols_to_remove)
    return dataset.remove_columns(cols_to_remove)
