from vod_benchmarking.models import (
    BenchmarkSpecificationSingle,
    BenchmarkSpecificationsBatch,
    IndexParameters,
    BenchmarkingResults,
)
from vod_benchmarking.benchmarking_datasets import DatasetHDF5Simple
from vod_benchmarking.docker_stats_logger import DockerMemoryLogger
from vod_search.base import SearchMaster

import numpy as np
from sklearn.neighbors import NearestNeighbors
from time import perf_counter, sleep
from typing import List, Type
import datetime
import traceback
from rich.progress import track
import pandas as pd
import os
from datetime import datetime
from loguru import logger
import time
import signal
import subprocess


class Timer:
    def __init__(self) -> None:
        self.t0: float = 0
        self.t1: float = 0
        self.timings: List[float] = []

    def begin(self) -> None:
        self.t0 = perf_counter()

    def end(self) -> None:
        self.t1 = perf_counter()
        self.timings.append(self.t1 - self.t0)

    @property
    def mean(self) -> float:
        return float(np.mean(self.timings))

    def pk_latency(self, k) -> float:
        return np.percentile(self.timings, k)

    def __str__(self) -> str:
        return f"{self.mean}s"


def get_ground_truth(vectors: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    """use sklearn to return flat, brute top_k NN indices"""
    nbrs = NearestNeighbors(n_neighbors=top_k, algorithm="brute").fit(vectors)  # type: ignore
    _, indices = nbrs.kneighbors(query)
    return indices


def recall_batch(index1: np.ndarray, index2: np.ndarray) -> float:
    def recall1d(index1: np.ndarray, index2: np.ndarray) -> float:
        return len(np.intersect1d(index1, index2)) / len(index1)

    _r = [recall1d(index1[i], index2[i]) for i in range(len(index1))]
    return np.array(_r).mean()


def recall_at_k(indices_pred: np.ndarray, indices_true: np.ndarray, k: int) -> float:
    def recall_at_k1d(indices_pred: np.ndarray, indices_true: np.ndarray, k: int) -> float:
        return float(np.isin(indices_true[0], indices_pred[:k]))

    # loop over the batch
    _r = [recall_at_k1d(indices_pred[i], indices_true[i], k) for i in range(len(indices_pred))]
    return np.array(_r).mean()


def create_index_parameters(label: str, index_types: List, preprocessings: List, metrics: List) -> List:
    _all_index_param = []

    for index_type in index_types:
        for preprocessing in preprocessings:
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


def timeout_handler(signum, frame):
    # timerIndexBuild.end()
    # print(f"Exact time duration: {timerIndexBuild.mean}")
    raise TimeoutError(f"Benchmarking trial timeout occurred")


def stop_docker_containers():
    print("Stopping all docker processes...")

    subprocess.run(["docker", "compose", "down", "-v"])
    sleep(5)
    subprocess.run(["docker", "network", "prune", "--force"])
    sleep(1)
    subprocess.run(
        [
            "docker",
            "stop",
            *subprocess.run(["docker", "ps", "-aq"], capture_output=True, text=True).stdout.split(),
        ]
    )
    sleep(1)
    subprocess.run(["sudo", "rm", "-rf", "volumes"])


def run_benchmark(bs: BenchmarkSpecificationSingle) -> BenchmarkingResults:
    dockerMemoryLogger = None
    try:
        # init
        TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # init. some variables
        os.environ["MKL_THREADING_LAYER"] = "TBB"
        benchmark_results = []

        tb = ""

        # timers
        benchmarkTimer = Timer()
        timerServerStartup = Timer()
        timerSearch = Timer()

        # prepare
        stop_docker_containers()  # stop all docker processes preemptively
        benchmarkTimer.begin()  # begin timer for this single benchmark process

        # get data
        dataset = bs.datasetClass()
        index_vectors, query_vectors = dataset.get_indices_and_queries_split(
            bs.n_query_vectors,
            bs.batch_size,
        )

        # begin logging memory consumption
        dockerMemoryLogger = DockerMemoryLogger(
            timestamp=TIMESTAMP,
            timeout=bs.timeout_benchmark,
        )

        # set alarm for index building timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(bs.timeout_index_build)

        # begin timing server startup
        timerServerStartup.begin()

        print("Spinning up server...")
        with bs.indexProviderClass(
            vectors=index_vectors,
            index_parameters=bs.indexParameters,
            dockerMemoryLogger=dockerMemoryLogger,
        ) as master:
            timerServerStartup.end()
            signal.alarm(0)  # Disable the timeout alarm

            # get client
            client = master.get_client()

            # warm up client
            for i in track(range(bs.n_warmup_batches), description=f"Warming up"):
                results = client.search(vector=query_vectors[i], top_k=bs.query_top_k_results)

            # define lists for results
            recalls = []
            recalls_at_1 = []
            recalls_at_10 = []
            recalls_at_100 = []
            recalls_at_1000 = []

            dockerMemoryLogger.set_begin_benchmarking()

            # begin benchmarking
            for i in track(range(bs.n_test_batches), description=f"Benchmarking"):
                i += bs.n_warmup_batches  # skip vectors used for warmup

                # get search results
                timerSearch.begin()
                results = client.search(vector=query_vectors[i], top_k=bs.query_top_k_results)
                timerSearch.end()

                indices_pred = results.indices

                # get true results
                indices_true = get_ground_truth(index_vectors, query_vectors[i], top_k=bs.query_top_k_results)

                # save recalls
                recalls.append(recall_batch(indices_pred, indices_true))
                recalls_at_1.append(recall_at_k(indices_pred, indices_true, 1))
                recalls_at_10.append(recall_at_k(indices_pred, indices_true, 10))
                recalls_at_100.append(recall_at_k(indices_pred, indices_true, 100))
                recalls_at_1000.append(recall_at_k(indices_pred, indices_true, 1000))

            # wrap up and return results
            benchmarkTimer.end()

            # stop logging
            dockerMemoryLogger.set_done_benchmarking()
            dockerMemoryLogger.stop_logging()

            # get memory logs
            memoryLogsBaseline, memoryLogsIngesting, memoryLogsBenchmark = dockerMemoryLogger.get_statistics()

            # get mean of recalls
            recall = float(np.mean(recalls))
            recall_at_1 = float(np.mean(recalls_at_1))
            recall_at_10 = float(np.mean(recalls_at_10))
            recall_at_100 = float(np.mean(recalls_at_100))
            recall_at_1000 = float(np.mean(recalls_at_1000))

            return BenchmarkingResults(
                bs,
                master.timerBuildIndex.mean,
                timerServerStartup.mean,
                timerSearch.mean,
                recall,
                recall_at_1,
                recall_at_10,
                recall_at_100,
                recall_at_1000,
                memoryLogsBaseline.mean() if memoryLogsBaseline.size > 0 else -1,
                memoryLogsIngesting.mean() if memoryLogsIngesting.size > 0 else -1,
                memoryLogsBenchmark.mean() if memoryLogsBenchmark.size > 0 else -1,
                memoryLogsBaseline,
                memoryLogsIngesting,
                memoryLogsBenchmark,
                timerSearch.timings,
                "",
            )

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        if dockerMemoryLogger:
            dockerMemoryLogger.stop_logging()
        # return -1 for all values except bs and error
        return BenchmarkingResults(
            bs, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, np.array([-1]), np.array([-1]), np.array([-1]), [-1], tb
        )
