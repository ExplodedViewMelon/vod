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

benchmarkSpecificationsBatch = [
    BenchmarkSpecificationsBatch(
        label="Faiss_PQ8_Optimization",
        indexProviderClasses=[
            faiss_search.FaissMaster,
        ],
        datasetClasses=[
            DatasetSift1M,
        ],
        indexTypes=[
            IVF(n_partition=256, n_probe=32),
        ],
        preprocessings=[
            ProductQuantization(m=16),
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="testBatch",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            qdrant_search.QdrantSearchMaster,
            milvus_search.MilvusSearchMaster,
        ],
        datasetClasses=[
            DatasetSift1M,
        ],
        indexTypes=[
            IVF(n_partition=256, n_probe=32),
            HNSW(M=16, ef_construction=32, ef_search=128),
        ],
        preprocessings=[
            None,
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
]

for benchmarkSpecifications in benchmarkSpecificationsBatch:
    print(f"Running batch: {benchmarkSpecifications.label}")
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
