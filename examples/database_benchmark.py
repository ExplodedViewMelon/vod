from __future__ import annotations

from vod_search import qdrant_local_search
import numpy as np
from sklearn.neighbors import NearestNeighbors

from time import perf_counter


def get_ground_truth(vectors: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    """use sklearn to return flat / brute top_k NN indices"""
    nbrs = NearestNeighbors(n_neighbors=top_k, algorithm="brute").fit(vectors)
    distances, indices = nbrs.kneighbors(query)
    return indices


def recall_batch(index1: np.ndarray, index2: np.ndarray) -> float:
    def recall1d(index1: np.ndarray, index2: np.ndarray) -> float:
        return len(np.intersect1d(index1, index2)) / len(index1)

    _r = [recall1d(index1[i], index2[i]) for i in range(len(index1))]
    return np.array(_r).mean()


class Timer:
    def __init__(self) -> None:
        self.t0: float = 0
        self.t1: float = 0
        self.durations = []

    def begin(self) -> None:
        self.t0 = perf_counter()

    def end(self) -> None:
        self.t1 = perf_counter()
        self.durations.append(self.t1 - self.t0)

    @property
    def duration(self) -> float:
        return np.mean(self.durations)

    def __str__(self) -> str:
        return f"{self.duration}s"


top_k = 100
database_size: int = 15_000
batch_size: int = 100
vector_size: int = 128
database_vectors = np.random.random(size=(database_size, vector_size))
query_vectors = np.random.random(size=(3, vector_size))

_Masters = [qdrant_local_search.QdrantLocalSearchMaster]

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()

benchmarkTimer.begin()

for _Master in _Masters:
    masterTimer.begin()
    with _Master(vectors=database_vectors) as master:
        masterTimer.end()

        client = master.get_client()

        searchTimer.begin()
        results = client.search(vector=query_vectors, top_k=top_k)
        searchTimer.end()

        pred_indices = results.indices
        true_indices = get_ground_truth(database_vectors, query_vectors, top_k=top_k)
        # print(pred_indices)
        # print(true_indices)

        print("Index build duration", masterTimer)
        print("Search duration", searchTimer)
        print("Recall:", recall_batch(pred_indices, true_indices))

benchmarkTimer.end()

print("Total time elapsed during bechmarking:", benchmarkTimer)
