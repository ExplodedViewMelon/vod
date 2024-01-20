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

    def __init__(self, *, m) -> None:
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
        assert M > 0
        assert ef_construction > 0
        assert ef_search > 0
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
        return f"{self.name}, n_partition={self.n_partition}"


class IndexParameters(abc.ABC):
    preprocessing: None | ProductQuantization | ScalarQuantization
    index_type: IVF | HNSW
    metric: str

    def __init__(self, *, preprocessing, index_type, metric, top_k) -> None:
        self.preprocessing = preprocessing
        self.index_type = index_type
        self.metric = metric

        if isinstance(self.preprocessing, ProductQuantization):
            assert isinstance(self.index_type, IVF), "HSNW does not support ProductQuantization"

    def __repr__(self) -> str:
        return f"{self.index_type}, {self.preprocessing}, {self.metric}"
