from __future__ import annotations
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
add databases to benchmark loop.
take care of parameter defaults in e.g. qdrant.
save the results in a document.
generate plots with distributions etc.
implement filtering and categories/subsets.
"""


class QdrantWrapper(qdrant_search.QdrantSearchMaster):
    def __init__(self, vectors: ndarray, index_specification: IndexSpecification, port=8888) -> None:
        assert index_specification.index[:4] == "HNSW"  # only index supported
        assert index_specification.distance == "DOT"  # hardcoded in client.py
        qdrant_body = {
            "shard_number": 1,
            "hnsw_config": {
                "ef_construct": 256,
                "m": index_specification.m,
            },
        }
        if index_specification.scalar_quantization:
            qdrant_body["quantization_config"] = {
                "scalar": {
                    "type": f"int{index_specification.scalar_quantization}",
                    "quantile": 0.99,
                    "always_ram": True,
                },
            }

        search_params = {
            "hnsw_ef": 256,
        }

        super().__init__(vectors=vectors, port=port, qdrant_body=qdrant_body, search_params=search_params)

    def __repr__(self) -> str:
        return "QdrantSearchMaster"


class FaissWrapper(faiss_search.FaissMaster):
    def __init__(self, vectors: ndarray, index_specification: IndexSpecification, port=8888):
        self.vectors = vectors
        self.index_specification = (index_specification,)
        self.port = port

    def __enter__(self):
        # build index for faiss search master
        # factory_string = "IVFauto,Flat"
        factory_string = self.convert_index_specification_to_faiss_factory_string(index_specification)
        if index_specification.distance == "INNER":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif index_specification.distance in ("L2", "EUCLID", "DOT"):
            faiss_metric = faiss.METRIC_L2  # TODO normalize if 'DOT'

        index = faiss_search.build_faiss_index(vectors=self.vectors, factory_string=factory_string, faiss_metric=faiss_metric)  # type: ignore
        # save into temp. folder
        self.tmpdir = tempfile.TemporaryDirectory()
        index_path = f"{self.tmpdir.name}/index.faiss"
        faiss.write_index(index, index_path)

        # continue with faiss search master
        super().__init__(index_path=index_path, port=self.port)
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        print("Cleaning up temp. folder")
        self.tmpdir.cleanup()  # clean up the temporary folder
        return super().__exit__(exc_type, exc_val, exc_tb)  # pass __exit__ onto faiss

    def convert_index_specification_to_faiss_factory_string(self, index_specification: IndexSpecification) -> str:
        factory_string = f"{index_specification.index}{index_specification.m}"
        if index_specification.scalar_quantization:
            factory_string += f"_SQ{index_specification.scalar_quantization}"
        return factory_string

    def __repr__(self) -> str:
        return "FaissSearchMaster"


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

_SearchMasters = [
    FaissWrapper,
    QdrantWrapper,
    # qdrant_local_search.QdrantLocalSearchMaster,
    # faiss_search.FaissMaster,
    # milvus_search.MilvusSearchMaster,
]

index_specifications = [
    IndexSpecification(index="HNSW", m=32, distance="DOT", scalar_quantization=8),
    IndexSpecification(index="HNSW", m=64, distance="DOT", scalar_quantization=8),
    IndexSpecification(index="HNSW", m=64, distance="DOT"),
]

# timers
benchmarkTimer = Timer()
masterTimer = Timer()
searchTimer = Timer()

benchmark_results = []

benchmarkTimer.begin()
for _SearchMaster in _SearchMasters:
    for index_specification in index_specifications:
        sleep(5)  # wait for server to terminate before creating new
        print("Spinning up server...")
        used_memory_before = psutil.virtual_memory().used
        masterTimer.begin()  # time server startup and build
        with _SearchMaster(vectors=index_vectors, index_specification=index_specification, port=8888) as master:
            client = master.get_client()
            masterTimer.end()
            used_memory_after = psutil.virtual_memory().used

            recalls = []
            recalls_at_1 = []
            recalls_at_10 = []
            recalls_at_100 = []

            for trial in track(range(n_warmup), description=f"Warming up {master.__repr__()}"):
                results = client.search(vector=query_vectors[trial], top_k=top_k)

            for trial in track(range(n_trials), description=f"Benchmarking {master.__repr__()}"):
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
                    "Index spec.": index_specification.index,
                    "Index dist.": index_specification.distance,
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

benchmarkTimer.end()

print("Total time elapsed during bechmarking:", benchmarkTimer)
print(pd.DataFrame(benchmark_results))
