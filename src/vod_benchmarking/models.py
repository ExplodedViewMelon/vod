from vod_benchmarking.benchmarking_datasets import DatasetHDF5Simple
import pydantic
import abc
from enum import Enum
from typing import List, Sequence, Type
import numpy as np
import pandas as pd


class ProductQuantization:
    m: int

    def __init__(self, *, m) -> None:
        """m must be 8 or 16"""
        self.m = m

    def __repr__(self) -> str:
        return f"PQ{self.m}"


class ScalarQuantization:
    n: int

    def __init__(self, *, n) -> None:
        self.n = n

    def __repr__(self) -> str:
        return f"SQ{self.n}"


class HNSW:
    name: str = "HNSW"
    M: int
    ef_construction: int
    ef_search: int  # search

    def __init__(self, *, M, ef_construction, ef_search) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def __repr__(self) -> str:
        return f"{self.name}, M={self.M}, ef_construction={self.ef_construction}, ef_search={self.ef_search}"


class IVF:
    name: str = "IVF"
    n_partition: int
    n_probe: int  # search

    def __init__(self, *, n_partition, n_probe) -> None:
        self.n_partition = n_partition
        self.n_probe = n_probe

    def __repr__(self) -> str:
        return f"{self.name}, n_partition={self.n_partition}, n_probe={self.n_probe}"


class IndexParameters(abc.ABC):
    preprocessing: None | ProductQuantization | ScalarQuantization
    index_type: IVF | HNSW
    metric: str

    def __init__(self, *, preprocessing, index_type, metric, top_k) -> None:
        self.preprocessing = preprocessing
        self.index_type = index_type
        self.metric = metric

    def __repr__(self) -> str:
        return f"{self.index_type}"


class DistanceMetric(Enum):
    L2 = "L2"
    DOT = "DOT"


class BenchmarkSpecificationSingle:
    def __init__(
        self,
        label: str,
        indexProviderClass,  #: Type[FaissMaster | QdrantSearchMaster | MilvusSearchMaster],
        datasetClass: Type[DatasetHDF5Simple],
        indexParameters: IndexParameters,
        batch_size: int,
        n_test_batches: int,
        query_top_k_results: int,
        timeout_index_build: int,
        timeout_benchmark: int,
    ):
        self.label = label
        self.datasetClass = datasetClass
        self.indexProviderClass = indexProviderClass
        self.indexParameters = indexParameters
        self.query_top_k_results = query_top_k_results
        self.n_test_batches = n_test_batches
        self.batch_size = batch_size
        self.timeout_index_build = timeout_index_build
        self.timeout_benchmark = timeout_benchmark
        # infer
        self.n_warmup_batches = self.n_test_batches // 5  # warmup using one fifth of training data size
        self.n_query_vectors = self.n_warmup_batches + self.n_test_batches  # total number of non-train vectors

    def get_aux_parameters(self) -> str:
        data = {
            "batch_size": self.batch_size,
            "n_test_batches": self.n_test_batches,
            "query_top_k_results": self.query_top_k_results,
            "timeout_index_build": self.timeout_index_build,
            "timeout_benchmark": self.timeout_benchmark,
        }
        return str(data)

    def get_summary(self) -> str:
        return (
            f"Summary for {self.label}:\n"
            f"indexProviderClass={self.indexProviderClass.get_name()},\n"
            f"datasetClass={self.datasetClass!r},\n"
            f"indexParameters={self.indexParameters!r},\n"
            f"batch_size={self.batch_size},\n"
            f"n_test_batches={self.n_test_batches},\n"
            f"query_top_k_results={self.query_top_k_results},\n"
            f"timeout_index_build={self.timeout_index_build},\n"
            f"timeout_benchmark={self.timeout_benchmark},\n"
            f"n_warmup_batches={self.n_warmup_batches},\n"
            f"n_query_vectors={self.n_query_vectors}"
        )


class BenchmarkSpecificationsBatch:
    def __init__(
        self,
        label: str,
        indexProviderClasses,  #: List[Type[FaissMaster | QdrantSearchMaster | MilvusSearchMaster]],
        datasetClasses: Sequence[Type[DatasetHDF5Simple]],
        indexTypes: Sequence[IVF | HNSW],
        preprocessings: Sequence[None | ProductQuantization | ScalarQuantization],
        distanceMetrics: Sequence[DistanceMetric],
        batch_size: int = 1000,
        n_test_batches: int = 1,
        query_top_k_results: int = 100,
        timeout_index_build: int = 60 * 20,
        timeout_benchmark: int = 60 * 30,
    ):
        """
        Initialize a BenchmarkSpecifications instance.

        :param label: A string label for the benchmark run.
        :param indexProviderClasses: A list of SearchMaster classes that provide indexing functionality.
        :param datasetClasses: A list of DatasetHDF5Simple classes representing the datasets to be used.
        :param indexTypes: A list of index types, which can be either IVF or HNSW.
        :param preprocessings: A list of preprocessing types, which can be either ProductQuantization or ScalarQuantization.
        :param distanceMetrics: A list of DistanceMetric enums representing the distance metrics to be used.
        :param batch_size: An integer representing the number of query vectors to process in a single batch.
        :param n_test_batches: An integer representing the number of test batches to process.
        :param query_top_k_results: An integer representing the number of top results to return for each query.
        :param timeout_index_build: An integer representing the maximum time in seconds allowed for building the index.
        """

        self.label = label
        self.datasetClasses = datasetClasses
        self.indexProviderClasses = indexProviderClasses
        self.indexTypes = indexTypes
        self.preprocessings = preprocessings
        self.distanceMetrics = distanceMetrics
        self.query_top_k_results = query_top_k_results
        self.n_test_batches = n_test_batches
        self.batch_size = batch_size
        self.timeout_index_build = timeout_index_build
        self.timeout_benchmark = timeout_benchmark

    def length(self):
        """
        Calculate the total number of benchmark specifications.

        :return: The total number of combinations of dataset classes, index provider classes, index types,
                 preprocessings, and distance metrics.
        """
        return (
            len(self.datasetClasses)
            * len(self.indexProviderClasses)
            * len(self.indexTypes)
            * len(self.preprocessings)
            * len(self.distanceMetrics)
        )

    def __iter__(self):
        """
        Yield each combination of the specified parameters
        """
        for datasetClass in self.datasetClasses:
            for indexProviderClass in self.indexProviderClasses:
                for indexType in self.indexTypes:
                    for preprocessing in self.preprocessings:
                        for distance_metric in self.distanceMetrics:
                            # make index parameter object to be passed to searchmaster
                            indexParameters = IndexParameters(
                                index_type=indexType,
                                metric=distance_metric.value,  # unpack enum
                                preprocessing=preprocessing,
                                top_k=self.query_top_k_results,
                            )
                            yield BenchmarkSpecificationSingle(
                                indexProviderClass=indexProviderClass,
                                datasetClass=datasetClass,
                                indexParameters=indexParameters,
                                # constants
                                label=self.label,
                                batch_size=self.batch_size,
                                n_test_batches=self.n_test_batches,
                                query_top_k_results=self.query_top_k_results,
                                timeout_index_build=self.timeout_index_build,
                                timeout_benchmark=self.timeout_benchmark,
                            )


class BenchmarkingResults:
    def __init__(
        self,
        benchmarkSpecification: BenchmarkSpecificationSingle,
        timingBuildIndex: float,
        timingServerStartup: float,
        timerBatchSearchMean: float,
        recall: float,
        recall_at_1: float,
        recall_at_10: float,
        recall_at_100: float,
        recall_at_1000: float,
        memoryBenchmark: float,
        memoryIngesting: float,
        memoryBaseline: float,
        allMemoryLogsBenchmark: np.ndarray,
        allMemoryLogsIngesting: np.ndarray,
        allMemoryLogsBaseline: np.ndarray,
        allTimingsSearch: List[float],
        error: str,
        aux: str,
    ):
        self.benchSpecification = benchmarkSpecification
        self.timingBuildIndex = timingBuildIndex
        self.timingServerStartup = timingServerStartup
        self.timerBatchSearchMean = timerBatchSearchMean
        self.recall = recall
        self.recall_at_1 = recall_at_1
        self.recall_at_10 = recall_at_10
        self.recall_at_100 = recall_at_100
        self.recall_at_1000 = recall_at_1000
        self.memoryBenchmark = memoryBenchmark
        self.memoryIngesting = memoryIngesting
        self.memoryBaseline = memoryBaseline
        self.allMemoryLogsBenchmark = allMemoryLogsBenchmark
        self.allMemoryLogsIngesting = allMemoryLogsIngesting
        self.allMemoryLogsBaseline = allMemoryLogsBaseline
        self.allTimingsSearch = allTimingsSearch
        self.error = error
        self.aux = aux

    def to_pandas(self) -> pd.DataFrame:
        data = {
            "label": [self.benchSpecification.label],
            "error": [self.error[-20:]],
            "indexProvider": [self.benchSpecification.indexProviderClass.get_name()],
            "indexType": [self.benchSpecification.indexParameters.index_type],
            # disable black formatter for the following lines
            # fmt: off
            "M": [self.benchSpecification.indexParameters.index_type.M if hasattr(self.benchSpecification.indexParameters.index_type, 'M') else None], #type:ignore
            "efSearch": [self.benchSpecification.indexParameters.index_type.ef_search if hasattr(self.benchSpecification.indexParameters.index_type, 'ef_search') else None], #type:ignore
            "efConstruction": [self.benchSpecification.indexParameters.index_type.ef_construction if hasattr(self.benchSpecification.indexParameters.index_type, 'ef_construction') else None], #type:ignore
            "nPartitions": [self.benchSpecification.indexParameters.index_type.n_partition if hasattr(self.benchSpecification.indexParameters.index_type, 'n_partition') else None], #type:ignore
            "nProbe": [self.benchSpecification.indexParameters.index_type.n_probe if hasattr(self.benchSpecification.indexParameters.index_type, 'n_probe') else None], #type:ignore
            "preprocessing": [self.benchSpecification.indexParameters.preprocessing],
            # fmt: on
            "distanceMetric": [self.benchSpecification.indexParameters.metric],
            "timingBuildIndex": [self.timingBuildIndex],
            "timingServerStartup": [self.timingServerStartup],
            "timingsSearch": [self.timerBatchSearchMean],
            "recall": [self.recall],
            "recall_at_1": [self.recall_at_1],
            "recall_at_10": [self.recall_at_10],
            "recall_at_100": [self.recall_at_100],
            "recall_at_1000": [self.recall_at_1000],
            "memoryBenchmark": [self.memoryBenchmark],
            "memoryIngesting": [self.memoryIngesting],
            "memoryBaseline": [self.memoryBaseline],
            "allMemoryLogsBenchmark": [self.allMemoryLogsBenchmark.tolist()],
            "allMemoryLogsIngesting": [self.allMemoryLogsIngesting.tolist()],
            "allMemoryLogsBaseline": [self.allMemoryLogsBaseline.tolist()],
            "allTimingsSearch": [self.allTimingsSearch],
            "fullError": [self.error],
            "aux": [self.aux],
        }
        return pd.DataFrame(data)
