import warnings

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
from datetime import datetime


benchmarkSpecificationsBatch = [
    BenchmarkSpecificationsBatch(
        label="Compare_Faiss_IVF_PQ",
        indexProviderClasses=[
            faiss_search.FaissMaster,
        ],
        datasetClasses=[
            DatasetSift1M,
        ],
        indexTypes=[
            IVF(n_partition=1, n_probe=1),
            IVF(n_partition=5, n_probe=1),
            IVF(n_partition=10, n_probe=1),
        ],
        preprocessings=[
            # None,
            ProductQuantization(m=8),
            # ProductQuantization(m=16),
        ],
        distanceMetrics=[
            DistanceMetric.L2,
        ],
    ),
    # BenchmarkSpecificationsBatch(
    #     label="testAllParameters",
    #     indexProviderClasses=[
    #         milvus_search.MilvusSearchMaster,
    #         faiss_search.FaissMaster,
    #         qdrant_search.QdrantSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[
    #         IVF(n_partition=256, n_probe=32),
    #         HNSW(M=16, ef_construction=32, ef_search=128),
    #     ],
    #     preprocessings=[
    #         None,
    #         ProductQuantization(m=8),
    #         ScalarQuantization(n=8),
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #         DistanceMetric.DOT,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_IVF_nPartitions",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[
    #         IVF(n_partition=2**9, n_probe=1),  # 512
    #         IVF(n_partition=2**10, n_probe=1),
    #         IVF(n_partition=2**11, n_probe=1),
    #         IVF(n_partition=2**12, n_probe=1),
    #         IVF(n_partition=2**13, n_probe=1),  # 8192
    #     ],
    #     preprocessings=[
    #         None,
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_IVF_nProbe",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[
    #         IVF(n_partition=1024, n_probe=1),
    #         IVF(n_partition=1024, n_probe=10),
    #         IVF(n_partition=1024, n_probe=100),
    #         IVF(n_partition=1024, n_probe=1000),
    #     ],
    #     preprocessings=[
    #         None,
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_PQ",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #         qdrant_search.QdrantSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[
    #         IVF(n_partition=1024, n_probe=100),
    #         HNSW(M=16, ef_construction=32, ef_search=64),
    #     ],
    #     preprocessings=[
    #         None,
    #         ProductQuantization(m=8),
    #         ProductQuantization(m=16),  # might cause benchmark to be killed
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_SQ",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #         qdrant_search.QdrantSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[
    #         IVF(n_partition=1024, n_probe=100),
    #         HNSW(M=16, ef_construction=32, ef_search=64),
    #     ],
    #     preprocessings=[
    #         None,
    #         ScalarQuantization(n=4),
    #         ScalarQuantization(n=8),
    #         ScalarQuantization(n=16),
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_HNSW_M",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #         qdrant_search.QdrantSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[HNSW(M=2**i, ef_construction=32, ef_search=64) for i in range(3, 9)],
    #     preprocessings=[
    #         None,
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_HNSW_efConstruction",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #         qdrant_search.QdrantSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[HNSW(M=16, ef_construction=2**i, ef_search=64) for i in range(3, 9)],
    #     preprocessings=[
    #         None,
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
    # BenchmarkSpecificationsBatch(
    #     label="Sweep_HNSW_efSearch",
    #     indexProviderClasses=[
    #         faiss_search.FaissMaster,
    #         milvus_search.MilvusSearchMaster,
    #         qdrant_search.QdrantSearchMaster,
    #     ],
    #     datasetClasses=[
    #         DatasetSift1M,
    #     ],
    #     indexTypes=[HNSW(M=16, ef_construction=32, ef_search=2**i) for i in range(3, 9)],
    #     preprocessings=[
    #         None,
    #     ],
    #     distanceMetrics=[
    #         DistanceMetric.L2,
    #     ],
    # ),
]

# make folder for results
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_directory = f"{os.getcwd()}/benchmarking_results/{current_time}"
os.makedirs(output_directory, exist_ok=True)

# make dataframe for all results, across batches
all_results = pd.DataFrame()

# loop over benchmark batches
for benchmarkSpecifications in benchmarkSpecificationsBatch:
    output_file = f"{output_directory}/{benchmarkSpecifications.label}.csv"
    print(f"Saving results into {output_file}")
    benchmarkingResultsAll = pd.DataFrame()

    # loop over individual benchmarks
    for benchmarkSpecification in benchmarkSpecifications:
        print(benchmarkSpecification.get_summary())
        benchmarkResults = run_benchmark(benchmarkSpecification).to_pandas()
        benchmarkingResultsAll = pd.concat([benchmarkingResultsAll, benchmarkResults], ignore_index=True)

    benchmarkingResultsAll.to_csv(output_file)
    print(benchmarkingResultsAll)

    all_results = pd.concat([all_results, benchmarkingResultsAll], ignore_index=True)

# Save the all_results DataFrame as a CSV file in the output directory
all_results_file = f"{output_directory}/all_results.csv"
all_results.to_csv(all_results_file)
