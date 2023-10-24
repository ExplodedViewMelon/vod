from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from time import perf_counter
from rich.progress import track
import pandas as pd

from vod_search import qdrant_local_search, milvus_search
from vod_datasets import DatasetGlove, DatasetLastFM, DatasetSift1M


def get_ground_truth(vectors: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    """use sklearn to return flat, brute top_k NN indices"""
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
    def mean(self) -> float:
        return np.mean(self.durations)

    def __str__(self) -> str:
        return f"{self.mean}s"


top_k = 100
n_trials = 100
n_warmup = 10
n_query_vectors = n_warmup + n_trials
query_batch_size = 10


dataset = DatasetLastFM()
index_vectors, query_vectors = dataset.get_indices_and_queries_split(n_query_vectors, query_batch_size)

_SearchMasters = [
    # faiss_search.FaissMaster,
    qdrant_local_search.QdrantLocalSearchMaster,
    # milvus_search.MilvusSearchMaster,
]

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()

benchmark_results = []

benchmarkTimer.begin()
for _SearchMaster in _SearchMasters:
    masterTimer.begin()
    with _SearchMaster(vectors=index_vectors) as master:
        masterTimer.end()

        client = master.get_client()

        recalls = []

        for trial in track(range(n_warmup), description=f"Warming up {master.__repr__()}"):
            results = client.search(vector=query_vectors[trial], top_k=top_k)

        for trial in track(range(n_trials), description=f"Benchmarking {master.__repr__()}"):
            searchTimer.begin()
            results = client.search(vector=query_vectors[trial + n_warmup], top_k=top_k)
            searchTimer.end()

            pred_indices = results.indices
            true_indices = get_ground_truth(index_vectors, query_vectors[trial + n_warmup], top_k=top_k)

            recalls.append(recall_batch(pred_indices, true_indices))

        benchmark_results.append(
            {
                "Index": master.__repr__(),
                "Build speed": masterTimer.mean,
                "Search speed avg.": searchTimer.mean,
                "Recall avg": np.mean(recalls),
            }
        )
        print("Index build duration", masterTimer.mean)
        print("Search duration", searchTimer.mean)
        print("Recall avg:", np.mean(recalls))

benchmarkTimer.end()

print("Total time elapsed during bechmarking:", benchmarkTimer)


print(pd.DataFrame(benchmark_results))
