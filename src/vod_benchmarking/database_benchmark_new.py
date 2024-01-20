from __future__ import annotations
import datetime
from pathlib import Path
from typing import Any
import numpy as np
from numpy import ndarray
import psutil
from sklearn.neighbors import NearestNeighbors
from time import perf_counter, sleep
from rich.progress import track
import pandas as pd
import faiss
from vod_search import qdrant_local_search, milvus_search, faiss_search, qdrant_search
from vod_search.models import IndexSpecification
from vod_benchmarking import DatasetGlove, DatasetLastFM, DatasetSift1M
import pydantic
import abc
import tempfile


"""
TODO
FIGURE OUT WHY FAISS DOES NOT UPDATE THE EF_ PARAMETERS

DONE - save the results in a document.
DONE - make the three master objects behave equally. Edit the template / base object.
add this to some computational setup
add databases to benchmark loop.
save all timers, plot histograms
implement filtering, categories, subsets etc.
a never ending task -> double check and polish the parameter specification. etc.

start writing theory for the report.
"""


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
        return float(np.mean(self.durations))

    def pk_latency(self, k) -> float:
        return np.percentile(self.durations, k)

    def __str__(self) -> str:
        return f"{self.mean}s"


from vod_search.models import *


def _create_index_param(preprocessings, index_types, metrics):
    _all_index_param = []

    for preprocessing in preprocessings:
        for index_type in index_types:
            for metric in metrics:
                _all_index_param.append(
                    IndexParameters(
                        preprocessing=preprocessing,
                        index_type=index_type,
                        metric=metric,
                        top_k=10,
                    )
                )
    return _all_index_param


top_k = 100
n_trials = 100
n_warmup = 10
n_query_vectors = n_warmup + n_trials
query_batch_size = 10

dataset = DatasetLastFM()
index_vectors, query_vectors = dataset.get_indices_and_queries_split(
    n_query_vectors,
    query_batch_size,  # size_limit=10_000
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

BENCHMARK_RUN_NAME = "Faiss_HNSW_ef_test"

_SearchMasters = [
    milvus_search.MilvusSearchMaster,
    faiss_search.FaissMaster,
    qdrant_search.QdrantSearchMaster,
]

preprocessings = [
    None,  # Remember this one!
    # ProductQuantization(m=5),  # must be divisible with n_dimensions
    # ProductQuantization(m=4),
    # ProductQuantization(m=8),
    # ScalarQuantization(n=8),
    # ScalarQuantization(n=8),
]

ef_parameter = 3  # like they recommended in the paper
index_types = [
    IVF(n_partition=500, n_probe=25),  # NOTE dim must be divisible with n_partition
    # IVF(n_partition=1000, n_probe=50),
    # IVF(n_partition=2000, n_probe=50),
    # HNSW(M=40, ef_construction=1 * d, ef_search=1 * d / 2),
    # HNSW(M=32, ef_construction=2, ef_search=16),
    HNSW(M=32, ef_construction=32, ef_search=16),
    # HNSW(M=32, ef_construction=128, ef_search=16),
    # HNSW(M=32, ef_construction=2, ef_search=64),
    # HNSW(M=32, ef_construction=32, ef_search=64),
    # HNSW(M=32, ef_construction=128, ef_search=64),
    # HNSW(M=160, ef_construction=ef_parameter * d, ef_search=ef_parameter * d),
    # NSW(M=13, ef_construction=10, ef_search=5),
]
metrics = [
    # "DOT",
    "L2",
]
index_specifications = _create_index_param(preprocessings, index_types, metrics)

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()

benchmark_results = []

n_benchmarks = len(_SearchMasters) * len(index_specifications)
print(f"Running {n_benchmarks} benchmarks in total.")
n = 0

benchmarkTimer.begin()
for _SearchMaster in _SearchMasters:
    for index_specification in index_specifications:
        n += 1
        # if np.random.random() > 5 / 60:  # should run for a minute then
        #     continue  # sample the benchmark
        print(f"Running {n} out of {n_benchmarks} benchmarks")
        master = None
        try:
            sleep(5)  # wait for server to terminate before creating new
            print("Spinning up server...")
            used_memory_before = psutil.virtual_memory().used
            masterTimer.begin()  # time server startup and build
            with _SearchMaster(vectors=index_vectors, index_parameters=index_specification) as master:
                print(f"Benchmarking {master}")

                client = master.get_client()
                masterTimer.end()
                used_memory_after = psutil.virtual_memory().used

                recalls = []
                recalls_at_1 = []
                recalls_at_10 = []
                recalls_at_100 = []

                for trial in track(range(n_warmup), description=f"Warming up"):
                    results = client.search(vector=query_vectors[trial], top_k=top_k)

                for trial in track(range(n_trials), description=f"Benchmarking"):
                    # get search results
                    searchTimer.begin()
                    results = client.search(vector=query_vectors[trial + n_warmup], top_k=top_k)
                    searchTimer.end()
                    pred_indices = results.indices

                    # get true results
                    true_indices = get_ground_truth(index_vectors, query_vectors[trial + n_warmup], top_k=top_k)

                    # save trial results
                    recalls.append(recall_batch(pred_indices, true_indices))
                    recalls_at_1.append(recall_at_k(pred_indices, true_indices, 1))
                    recalls_at_10.append(recall_at_k(pred_indices, true_indices, 10))
                    recalls_at_100.append(recall_at_k(pred_indices, true_indices, 100))

                benchmark_results.append(
                    {
                        "Index": master.__repr__(),
                        "Index parameters.": f"{index_specification}",
                        "Build speed (s)": masterTimer.mean,
                        "Search speed avg. (ms)": searchTimer.mean * 1000,
                        "Search speed p95 (ms)": searchTimer.pk_latency(95) * 1000,
                        "Recall avg": np.mean(recalls),
                        "Recall@1": np.mean(recalls_at_1),
                        "Recall@10": np.mean(recalls_at_10),
                        "Recall@100": np.mean(recalls_at_100),
                        "Est. memory usage (GB)": (used_memory_after - used_memory_before) / (1024**3),
                    }
                )
        except Exception as e:
            print("Benchmark went wrong.", e)
            benchmark_results.append(
                {
                    "Index": master.__repr__() if master else "None",
                    "Index parameters.": f"error: {e}",
                    "Build speed (s)": -1,
                    "Search speed avg. (ms)": -1,
                    "Search speed p95 (ms)": -1,
                    "Recall avg": -1,
                    "Recall@1": -1,
                    "Recall@10": -1,
                    "Recall@100": -1,
                    "Est. memory usage (GB)": -1,
                }
            )


benchmarkTimer.end()

print("Total time elapsed during bechmarking:", benchmarkTimer)
df_results = pd.DataFrame(benchmark_results)
print(df_results)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
path = "/Users/oas/Documents/VOD/vod/benchmarking_results"
output_file = f"{path}/{BENCHMARK_RUN_NAME}_{timestamp}.csv"
df_results.to_csv(output_file)
