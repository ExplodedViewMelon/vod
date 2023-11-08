from pydantic import BaseModel, model_validator
import pydantic


class InitializeIndexRequest(BaseModel):
    """Configuration used to init/build an index."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    index_path: str = pydantic.Field(...)

    # @model_validator(mode="after")
    # def check_cloner_options(self) -> "InitializeIndexRequest":
    #     """Validate cloner options."""
    #     if self.serve_on_gpu and not self.cloner_options:
    #         raise ValueError("`cloner_options` must be provided when `serve_on_gpu` is `True`")
    #     return self


class FaissInitConfig(BaseModel):
    """Configuration used to init/build a faiss index."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    index_path: str
    nprobe: int = 8


class InitResponse(BaseModel):
    """Response to the initialization request."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    success: bool
    exception: None | str


class SearchFaissQuery(BaseModel):
    """Query to search a faiss index."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    vectors: list = pydantic.Field(..., description="A batch of vectors. Implicitly defines `list[list[float]]`.")
    top_k: int = 3


class FastSearchFaissQuery(BaseModel):
    """This is the same as SearchFaissQuery, but with the vectors serialized."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    vectors: str = pydantic.Field(..., description="A batch of serialized vectors")
    top_k: int = 3


class FaissSearchResponse(BaseModel):
    """Response to the search request."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    scores: list = pydantic.Field(..., description="A batch of scores. Implicitly defines `list[list[float]]`.")
    indices: list = pydantic.Field(..., description="A batch of indices. Implicitly defines `list[list[int]]`.")


class FastFaissSearchResponse(BaseModel):
    """This is the same as FaissSearchResponse, but with the vectors serialized."""

    class Config:
        """pydantic config."""

        frozen = False
        extra = "forbid"

    scores: str
    indices: str
