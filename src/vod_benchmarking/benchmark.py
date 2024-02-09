from __future__ import annotations
import datetime
from pathlib import Path
import traceback
from typing import Any, Type
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
from vod_benchmarking import DatasetGlove, DatasetLastFM, DatasetSift1M, DatasetHDF5Simple
import pydantic
import abc
import tempfile
import os
from vod_search.models import *  # TODO import each thing instead of wildcard.
from vod_benchmarking import DockerMemoryLogger
from datetime import datetime
import os
from vod_benchmarking import Timer
import signal

os.environ["MKL_THREADING_LAYER"] = "TBB"


"""
TODO
DONE - save the results in a document.
DONE - make the three master objects behave equally. Edit the template / base object.
add databases to benchmark loop.
save all timers, plot histograms
implement filtering, categories, subsets etc.
a never ending task -> double check and polish the parameter specification. etc.
start writing theory for the report.
Fix folder structure.
Make build_index a standard thing in base object.
"""

# most important hyper parameters:


_SearchMasters = [
    faiss_search.FaissMaster,
    # milvus_search.MilvusSearchMaster,
    # qdrant_search.QdrantSearchMaster,
]

preprocessings = [
    None,  # Remember this one!
    # ProductQuantization(m=5),  # must be divisible with n_dimensions
    # ProductQuantization(m=4),
    # ProductQuantization(m=4),
    ProductQuantization(m=8),
    # ScalarQuantization(n=8),
    ScalarQuantization(n=8),
]

ef_parameter = 3  # like they recommended in the paper
index_types = [
    IVF(n_partition=500, n_probe=25),  # NOTE dim must be divisible with n_partition
    # IVF(n_partition=2000, n_probe=100),
    # IVF(n_partition=8000, n_probe=1000),
    # IVF(n_partition=16000, n_probe=1000),
    # IVF(n_partition=32000, n_probe=5000),
    # IVF(n_partition=1000, n_probe=50),
    # HNSW(M=40, ef_construction=1 * d, ef_search=1 * d / 2),
    # HNSW(M=32, ef_construction=2, ef_search=16),
    # HNSW(M=32, ef_construction=32, ef_search=16),
    # HNSW(M=32, ef_construction=128, ef_search=64),
    # HNSW(M=64, ef_construction=64, ef_search=64),
    # HNSW(M=64, ef_construction=256, ef_search=64),
    HNSW(M=64, ef_construction=256, ef_search=128),
    # HNSW(M=128, ef_construction=256, ef_search=256),
    # HNSW(M=256, ef_construction=256, ef_search=256),
    # HNSW(M=160, ef_construction=ef_parameter * d, ef_search=ef_parameter * d),
    # NSW(M=13, ef_construction=10, ef_search=5),
]
metrics = [
    # "DOT",
    "L2",
]
# datasets_classes: list[Type[DatasetHDF5Simple]] = [DatasetSift1M, DatasetGlove, DatasetLastFM]
datasets_classes: list[Type[DatasetHDF5Simple]] = [DatasetSift1M]


def get_ground_truth(vectors: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    """use sklearn to return flat, brute top_k NN indices"""
    nbrs = NearestNeighbors(n_neighbors=top_k, algorithm="brute").fit(vectors)  # type: ignore
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
n_trials = 300
n_warmup = n_trials // 5
n_query_vectors = n_warmup + n_trials
query_batch_size = 10


index_specifications = _create_index_param(preprocessings, index_types, metrics)

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()


def timeout_handler(signum, frame):
    # masterTimer.end()
    # print(f"Exact time duration: {masterTimer.mean}")
    raise TimeoutError(f"Timeout occurred ({TIMEOUT_INDEX_BUILD} s)")


benchmark_results = []
n = 0
dockerMemoryLogger = None

timestamps = []
tb = ""
benchmarkTimer.begin()

# TIMEOUT_INDEX_BUILD = 120  # seconds
TIMEOUT_INDEX_BUILD = 60 * 5  # seconds

n_benchmarks = len(datasets_classes) * len(_SearchMasters) * len(index_specifications)
print(f"Running {n_benchmarks} benchmarks in total.")
for dataset_class in datasets_classes:
    dataset: DatasetHDF5Simple = dataset_class()
    index_vectors, query_vectors = dataset.get_indices_and_queries_split(
        n_query_vectors,
        query_batch_size,  # size_limit=10_000
    )
    # _, d = index_vectors.shape

    print(f"Using data from {dataset}")
    print(
        "with",
        n_query_vectors * query_batch_size,
        "queries in",
        n_query_vectors,
        "batches of size",
        query_batch_size,
        "and shape",
        query_vectors.shape,
    )
    for _SearchMaster in _SearchMasters:
        for index_specification in index_specifications:
            n += 1
            # if np.random.random() > 5 / 60:  # should run for a minute then
            #     continue  # sample the benchmark
            run_id = str(np.random.random())[2:8]
            run_title = f"{_SearchMaster} {index_specification}. ID: {run_id}"
            print(f"Running {n} out of {n_benchmarks} benchmarks. Title: {run_title}")
            master = None
            try:
                sleep(5)  # wait for server to terminate before creating new
                print("Spinning up server...")

                dockerMemoryLogger = DockerMemoryLogger(title=run_id)
                masterTimer.begin()  # time server startup and build
                timestamps.append(("BeginServer", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                dockerMemoryLogger.set_begin_ingesting()

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TIMEOUT_INDEX_BUILD)  # Set timeout alarm
                with _SearchMaster(vectors=index_vectors, index_parameters=index_specification) as master:
                    masterTimer.end()
                    dockerMemoryLogger.set_done_ingesting()
                    signal.alarm(0)  # Disable the timeout alarm

                    print(f"Benchmarking {master}")

                    client = master.get_client()

                    recalls = []
                    recalls_at_1 = []
                    recalls_at_10 = []
                    recalls_at_100 = []

                    for trial in track(range(n_warmup), description=f"Warming up"):
                        results = client.search(vector=query_vectors[trial], top_k=top_k)

                    dockerMemoryLogger.set_begin_benchmarking()
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

                    dockerMemoryLogger.set_done_benchmarking()
                    dockerMemoryLogger.stop_logging()

                    memory_statistics: dict[str, float] = dockerMemoryLogger.get_statistics()

                    try:
                        dockerMemoryLogger.make_plots()
                    except Exception as pe:
                        print("Making plots went wrong.", pe)

                    benchmark_results.append(
                        {
                            "Dataset": dataset.__repr__(),
                            "Index": master.__repr__(),
                            "Index parameters.": f"{index_specification}",
                            "Server startup speed (s)": masterTimer.mean - master.timerBuildIndex.mean,
                            "Index build speed (s)": master.timerBuildIndex.mean,
                            "Search speed avg. (ms)": searchTimer.mean * 1000,
                            "Search speed p95 (ms)": searchTimer.pk_latency(95) * 1000,
                            "Recall avg": np.mean(recalls),
                            "Recall@1": np.mean(recalls_at_1),
                            "Recall@10": np.mean(recalls_at_10),
                            "Recall@100": np.mean(recalls_at_100),
                            **memory_statistics,
                        }
                    )
            except Exception:
                print("Benchmark went wrong.")
                tb = traceback.format_exc()
                print(tb)
                if dockerMemoryLogger:
                    dockerMemoryLogger.stop_logging()
                benchmark_results.append(
                    {
                        "Dataset": dataset.__repr__() if dataset else "None",
                        "Index": master.__repr__() if master else "None",
                        "Index parameters.": f"error: {tb}",
                        "Server startup speed (s)": -1,
                        "Index build speed (s)": -1,
                        "Search speed avg. (ms)": -1,
                        "Search speed p95 (ms)": -1,
                        "Recall avg": -1,
                        "Recall@1": -1,
                        "Recall@10": -1,
                        "Recall@100": -1,
                        "ingesting_max": -1,
                        "ingesting_mean": -1,
                        "benchmarking_max": -1,
                        "benchmarking_mean": -1,
                    }
                )


benchmarkTimer.end()

print("Total time elapsed during bechmarking:", benchmarkTimer)
df_results = pd.DataFrame(benchmark_results)
print(df_results)

timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
output_directory = f"{os.getcwd()}/benchmarking_results"
os.makedirs(output_directory, exist_ok=True)

output_file = f"{output_directory}/{timestamp}.csv"
print("saving results to", output_file)
df_results.to_csv(output_file)

print(timestamps)
