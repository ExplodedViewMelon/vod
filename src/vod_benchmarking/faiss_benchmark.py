from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors
from time import perf_counter, sleep
from rich.progress import track
import pandas as pd
import faiss
from vod_benchmarking import DatasetGlove, DatasetLastFM, DatasetSift1M


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


def recall_at_k(index1: np.ndarray, index2: np.ndarray, k: int) -> float:
    def recall_at_k1d(index1: np.ndarray, index2: np.ndarray, k: int) -> float:
        return float(np.isin(index2[0], index1[:k]))

    _r = [recall_at_k1d(index1[i], index2[i], k) for i in range(len(index1))]
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

    def pk_latency(self, k) -> float:
        return np.percentile(self.durations, k)

    def __str__(self) -> str:
        return f"{self.mean}s"


top_k = 100
n_trials = 1000
n_warmup = 100
n_query_vectors = n_warmup + n_trials
query_batch_size = 10


dataset = DatasetLastFM()  # TODO put this into the main loop.
index_vectors, query_vectors = dataset.get_indices_and_queries_split(
    n_query_vectors,
    query_batch_size,  # size_limit=100_000
)
n, d = index_vectors.shape
print(
    "Using",
    n_query_vectors * query_batch_size,
    "queries in",
    n_query_vectors,
    "batches of size",
    query_batch_size,
    "and shape",
    query_vectors.shape,
)

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()

benchmark_results = []

index_specifications = [
    "Flat",
    # "HNSW32",
    # "HNSW64",
    # "HNSW128",
    # "IVF256,Flat",
    # "IVF512,Flat",
    # "IVF256,PQ1",
    # "IVF256,PQ5",
    # "IVF256,PQ13",
    # "IVF512,PQ1",
    # "IVF512,PQ5",
    # "IVF512,PQ13",
    # "IVF4096_HNSW32,Flat",
    # "OPQ32,IVF4096_HNSW,PQ32",
    # "OPQ{d},IVF256,PQ{d}x8",
    # f"OPQ{d},IVF256,PQ{d},RFlat",
]  #

benchmarkTimer.begin()
for index_specification in index_specifications:
    print("Running benchmark for", index_specification)
    try:
        # build index
        masterTimer.begin()
        index_faiss = faiss.index_factory(d, index_specification)
        index_faiss.train(index_vectors)
        index_faiss.add(index_vectors)
        masterTimer.end()

        recalls = []
        recalls_at_1 = []
        recalls_at_10 = []
        recalls_at_100 = []

        for trial in track(range(n_warmup), description=f"Warming up"):
            results = index_faiss.search(query_vectors[trial], k=top_k)

        for trial in track(range(n_trials), description=f"Benchmarking"):
            # get search results
            searchTimer.begin()
            distances, pred_indices = index_faiss.search(query_vectors[trial + n_warmup], k=top_k)
            searchTimer.end()

            # get true results
            true_indices = get_ground_truth(index_vectors, query_vectors[trial + n_warmup], top_k=top_k)

            # save trial results
            recalls.append(recall_batch(pred_indices, true_indices))
            recalls_at_1.append(recall_at_k(pred_indices, true_indices, 1))
            recalls_at_10.append(recall_at_k(pred_indices, true_indices, 10))
            recalls_at_100.append(recall_at_k(pred_indices, true_indices, 100))

        # save test results
        benchmark_results.append(
            {
                "Index factory.": index_specification,
                "Build speed": masterTimer.mean,
                "Search speed avg. (ms)": searchTimer.mean * 1000,
                "Search speed p95 (ms)": searchTimer.pk_latency(95) * 1000,
                "Recall avg": np.mean(recalls),
                "Recall@1": np.mean(recalls_at_1),
                "Recall@10": np.mean(recalls_at_10),
                "Recall@100": np.mean(recalls_at_100),
            }
        )
    except Exception as e:
        print("An error occured:", e, "resuming...")


benchmarkTimer.end()

print("Total time elapsed during bechmarking:", benchmarkTimer)
print(pd.DataFrame(benchmark_results))
