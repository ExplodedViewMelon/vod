import numpy as np
import pytest

from raffle_ds_research.tools import c_tools


def _get_expected_labels(queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    buffer = np.full(queries.shape, np.nan, dtype=np.float32)
    for i, q in enumerate(queries):
        for j, k in enumerate(keys):
            if q == k:
                buffer[i] = values[j]
                break
    return buffer


@pytest.mark.parametrize("n_points", [10, 100])
@pytest.mark.parametrize("max_n_unique", [10, 100])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gather_by_index(n_points: int, max_n_unique: int, seed: int):
    np.random.set_state(np.random.RandomState(seed).get_state())
    queries = np.random.randint(0, max_n_unique, size=(n_points,), dtype=np.uint64)
    keys = np.random.randint(0, max_n_unique, size=(n_points,), dtype=np.uint64)
    values = np.random.random(size=(n_points,))

    c_result = c_tools.gather_by_index(queries, keys, values)
    py_result = _get_expected_labels(queries, keys, values)
    for i, (c, py) in enumerate(zip(c_result, py_result)):
        if not np.isclose(c, py) and not (np.isnan(c) and np.isnan(py)):
            raise ValueError(f"c_result[{i}] = {c} != {py} = py_result[{i}]")