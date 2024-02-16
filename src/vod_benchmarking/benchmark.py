"""
TODO
DONE - save the results in a document.
DONE - make the three master objects behave equally. Edit the template / base object.
DONE - add databases to benchmark loop.
save all timers, plot histograms
implement filtering, categories, subsets etc.
a never ending task -> double check and polish the parameter specification. etc.
start writing theory for the report.
Fix folder structure.
Make build_index a standard thing in base object.
start writing report...
Think about how you query e.g. number of vectors in database, number of queries etc. etc.
Add docker compose down / docker stop all containers before starting benchmark.
"""

from __future__ import annotations
import datetime
import traceback
from typing import Any, Type
import numpy as np
from time import perf_counter, sleep
from rich.progress import track
import pandas as pd
from vod_search import milvus_search, faiss_search, qdrant_search
from vod_benchmarking import DatasetGlove, DatasetLastFM, DatasetSift1M, DatasetHDF5Simple
import os
from vod_search.models import HNSW, IVF, ScalarQuantization, ProductQuantization
from vod_benchmarking import DockerMemoryLogger
from datetime import datetime
from vod_benchmarking import Timer
import signal
from loguru import logger
from vod_benchmarking.functions_benchmark import (
    timeout_handler,
    create_index_parameters,
    get_ground_truth,
    recall_batch,
    recall_at_k,
    stop_docker_containers,
)

os.environ["MKL_THREADING_LAYER"] = "TBB"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# HYPERPARAMETERS

_SearchMasters = [
    qdrant_search.QdrantSearchMaster,
    milvus_search.MilvusSearchMaster,
    faiss_search.FaissMaster,
]

preprocessings = [
    None,  # Remember this one!
    ProductQuantization(m=8),  # must be divisible with n_dimensions
    ScalarQuantization(n=8),
]

index_types = [
    IVF(n_partition=100, n_probe=100),
    IVF(n_partition=1000, n_probe=100),
    IVF(n_partition=10000, n_probe=100),
    HNSW(M=16, ef_construction=128, ef_search=128),
    HNSW(M=32, ef_construction=128, ef_search=128),
    HNSW(M=64, ef_construction=128, ef_search=128),
]
metrics = [
    "L2",
    "DOT",
]

datasets_classes: list[Type[DatasetHDF5Simple]] = [DatasetSift1M, DatasetGlove, DatasetLastFM]
# datasets_classes: list[Type[DatasetHDF5Simple]] = [DatasetGlove] # smallest
# datasets_classes: list[Type[DatasetHDF5Simple]] = [DatasetSift1M] # largest

top_k = 100
n_trials = 300
n_warmup = n_trials // 5
n_query_vectors = n_warmup + n_trials
query_batch_size = 10
TIMEOUT_INDEX_BUILD = 60 * 10  # seconds

index_specifications = create_index_parameters(preprocessings, index_types, metrics)

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()

# lists
benchmark_results = []

benchmark_counter = 0
dockerMemoryLogger = None
tb = ""

print("Stopping all docker containers preemptively")
stop_docker_containers()

number_of_benchmarks = len(datasets_classes) * len(_SearchMasters) * len(index_specifications)
print(f"Running {number_of_benchmarks} benchmarks in total.")

benchmarkTimer.begin()
for dataset_class in datasets_classes:
    dataset: DatasetHDF5Simple = dataset_class()
    index_vectors, query_vectors = dataset.get_indices_and_queries_split(
        n_query_vectors,
        query_batch_size,
    )

    print(
        f"Using data from {dataset}",
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
        # get search master name
        searchMasterName: str = "UnknownIndex"
        if _SearchMaster == qdrant_search.QdrantSearchMaster:
            searchMasterName = "Qdrant"
        elif _SearchMaster == faiss_search.FaissMaster:
            searchMasterName = "Faiss"
        elif _SearchMaster == milvus_search.MilvusSearchMaster:
            searchMasterName = "Milvus"

        for index_specification in index_specifications:
            benchmark_counter += 1
            run_title = f"{searchMasterName} {index_specification}. TIMESTAMP: {TIMESTAMP}"
            print(f"Running {benchmark_counter} out of {number_of_benchmarks} benchmarks. Title: {run_title}")
            master = None
            try:
                sleep(5)  # wait for server to terminate before creating new

                # begin logging
                dockerMemoryLogger = DockerMemoryLogger(
                    timestamp=TIMESTAMP, index_specification=f"{index_specification}", searchMasterName=searchMasterName
                )
                masterTimer.begin()
                dockerMemoryLogger.set_begin_ingesting()

                # set timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TIMEOUT_INDEX_BUILD)

                print("Spinning up server...")
                with _SearchMaster(vectors=index_vectors, index_parameters=index_specification) as master:
                    masterTimer.end()
                    dockerMemoryLogger.set_done_ingesting()
                    signal.alarm(0)  # Disable the timeout alarm

                    recalls = []
                    recalls_at_1 = []
                    recalls_at_10 = []
                    recalls_at_100 = []

                    client = master.get_client()

                    # warm up
                    for trial in track(range(n_warmup), description=f"Warming up"):
                        results = client.search(vector=query_vectors[trial], top_k=top_k)

                    # start benchmarking
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

                    # stop logging
                    dockerMemoryLogger.set_done_benchmarking()
                    dockerMemoryLogger.stop_logging()

                    # save stats
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

                print("Stopping all docker processes...")
                stop_docker_containers()

                benchmark_results.append(
                    {
                        "Dataset": dataset.__repr__() if dataset else "None",
                        "Index": master.__repr__() if master else f"{searchMasterName} {index_specification}",
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
df_results = pd.DataFrame(benchmark_results)
output_directory = f"{os.getcwd()}/benchmarking_results"
os.makedirs(output_directory, exist_ok=True)
output_file = f"{output_directory}/{TIMESTAMP}.csv"
df_results.to_csv(output_file)

print("Total time elapsed during bechmarking:", benchmarkTimer)
print("Saving results to", output_file)
