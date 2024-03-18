from vod_benchmarking.models import (
    HNSW,
    IVF,
    BenchmarkSpecificationSingle,
    ScalarQuantization,
    ProductQuantization,
    BenchmarkSpecificationsBatch,
    DistanceMetric,
)
from vod_benchmarking.functions_benchmark import run_benchmark
from vod_benchmarking.benchmarking_datasets import DatasetGlove, DatasetLastFM, DatasetSift1M, DatasetGIST
from vod_search import milvus_search, faiss_search, qdrant_search
from typing import List, Type
import traceback
import pandas as pd
import os

# SETUP

indexProviderClasses = [
    # faiss_search.FaissMaster,
    qdrant_search.QdrantSearchMaster,
    milvus_search.MilvusSearchMaster,
]

preprocessings = [
    None,  # Remember this one!
    # ProductQuantization(m=4),  # must be divisible with n_dimensions
    # ProductQuantization(m=8),  # i.e. 128 for sift
    # ProductQuantization(m=16),
    # ScalarQuantization(n=4),
    # ScalarQuantization(n=6),
    # ScalarQuantization(n=8),
]

indexTypes = [
    IVF(n_partition=256, n_probe=32),
    # IVF(n_partition=512, n_probe=64),
    # HNSW(M=8, ef_construction=16, ef_search=128),
    HNSW(M=16, ef_construction=32, ef_search=128),
]
distanceMetrics = [
    DistanceMetric.L2,
    # DistanceMetric.DOT,
]

datasetClasses = [
    # DatasetGlove,  # the angular one
    # DatasetLastFM,
    DatasetSift1M,
    # DatasetGIST,
]

benchmarkSpecifications = BenchmarkSpecificationsBatch(
    label="testBatch",
    indexProviderClasses=indexProviderClasses,
    datasetClasses=datasetClasses,  # type: ignore
    indexTypes=indexTypes,
    preprocessings=preprocessings,
    distanceMetrics=distanceMetrics,
)

benchmarkingResultsAll = pd.DataFrame()

# run benchmark
for benchmarkSpecification in benchmarkSpecifications:
    benchmarkResults = run_benchmark(benchmarkSpecification).to_pandas()
    # save parameters
    benchmarkingResultsAll = pd.concat([benchmarkingResultsAll, benchmarkResults], ignore_index=True)


output_directory = f"{os.getcwd()}/benchmarking_results"
os.makedirs(output_directory, exist_ok=True)
output_file = f"{output_directory}/{benchmarkSpecifications.label}.csv"
benchmarkingResultsAll.to_csv(output_file)


print("Results")
print(benchmarkingResultsAll)
