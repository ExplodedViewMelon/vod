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
    # BenchmarkSpecificationsBatch(
    #     label="simplestTest",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[
    #         IVF(n_partition=256, n_probe=32),
    #     ],
    #     preprocessings=[
    #         None,
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    BenchmarkSpecificationsBatch(
        label="testAllParameters",
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
            ProductQuantization(m=8),
            ScalarQuantization(n=8),
        ],
        distanceMetrics=[
            DistanceMetric.L2,
            DistanceMetric.DOT,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_IVF_nPartitions",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[
            IVF(n_partition=2**9, n_probe=1),  # 512
            IVF(n_partition=2**10, n_probe=1),
            IVF(n_partition=2**11, n_probe=1),
            IVF(n_partition=2**12, n_probe=1),
            IVF(n_partition=2**13, n_probe=1),  # 8192
        ],
        preprocessings=[
            None,
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_IVF_nProbe",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[
            IVF(n_partition=1024, n_probe=1),
            IVF(n_partition=1024, n_probe=10),
            IVF(n_partition=1024, n_probe=100),
            IVF(n_partition=1024, n_probe=1000),
        ],
        preprocessings=[
            None,
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_PQ",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
            qdrant_search.QdrantSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[
            IVF(n_partition=1024, n_probe=100),
            HNSW(M=16, ef_construction=32, ef_search=64),
        ],
        preprocessings=[
            None,
            ProductQuantization(m=4),
            ProductQuantization(m=8),
            ProductQuantization(m=16),
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_SQ",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
            qdrant_search.QdrantSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[
            IVF(n_partition=1024, n_probe=100),
            HNSW(M=16, ef_construction=32, ef_search=64),
        ],
        preprocessings=[
            None,
            ScalarQuantization(n=4),
            ScalarQuantization(n=8),
            ScalarQuantization(n=16),
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_HNSW_M",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
            qdrant_search.QdrantSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[HNSW(M=2**i, ef_construction=32, ef_search=64) for i in range(3, 9)],
        preprocessings=[
            None,
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_HNSW_efConstruction",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
            qdrant_search.QdrantSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[HNSW(M=16, ef_construction=2**i, ef_search=64) for i in range(3, 9)],
        preprocessings=[
            None,
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    BenchmarkSpecificationsBatch(
        label="Sweep_HNSW_efSearch",
        indexProviderClasses=[
            faiss_search.FaissMaster,
            milvus_search.MilvusSearchMaster,
            qdrant_search.QdrantSearchMaster,
        ],
        datasetClasses=[
            DatasetGIST,
        ],
        indexTypes=[HNSW(M=16, ef_construction=32, ef_search=2**i) for i in range(3, 9)],
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
        print("Running benchmark")
        benchmarkSpecification.print_summary()

        benchmarkResults = run_benchmark(benchmarkSpecification).to_pandas()
        # save parameters
        benchmarkingResultsAll = pd.concat([benchmarkingResultsAll, benchmarkResults], ignore_index=True)

    output_directory = f"{os.getcwd()}/benchmarking_results"
    os.makedirs(output_directory, exist_ok=True)
    output_file = f"{output_directory}/{benchmarkSpecifications.label}.csv"
    benchmarkingResultsAll.to_csv(output_file)

    print("Results")
    print(benchmarkingResultsAll)
