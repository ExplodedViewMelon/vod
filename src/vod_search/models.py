import pydantic
import abc


# V legacy
class IndexSpecification(pydantic.BaseModel):
    index: str = pydantic.Field(..., description="Index type e.g. HNSW")
    m: int = pydantic.Field(...)
    distance: str = pydantic.Field(...)
    scalar_quantization: int = pydantic.Field(default=None)
    # *,
    vectors_path: str = pydantic.Field(default="", description="A path pointing to saved vectors.")


class ProductQuantization:
    m: int
    n_bits: int

    def __init__(self, *, m, n_bits=8) -> None:
        self.m = m
        self.n_bits = n_bits


class ScalarQuantization:
    n: int

    def __init__(self, *, n) -> None:
        self.n = n


class HNSW:
    name: str = "HNSW"
    ef_construction: int
    M: int
    ef_search: int  # search

    def __init__(self, *, ef_construction, ef_search, M) -> None:
        assert ef_construction > 0
        assert ef_search > 0
        assert M > 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M


class IVF:
    name: str = "IVF"
    n_partition: int
    n_probe: int  # search

    def __init__(self, *, n_partition, n_probe) -> None:
        assert n_partition > 0
        assert n_probe > 0
        self.n_partition = n_partition
        self.n_probe = n_probe


class IndexParameters(abc.ABC):
    preprocessing: None | ProductQuantization | ScalarQuantization
    index_type: IVF | HNSW
    metric: str
    top_k: int

    def __init__(self, *, preprocessing, index_type, metric, top_k) -> None:
        self.preprocessing = preprocessing
        self.index_type = index_type
        self.metric = metric
        self.top_k = top_k
