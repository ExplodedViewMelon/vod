from pydantic import BaseModel, Field, model_validator


class InitializeIndexRequest(BaseModel):
    """Configuration used to init/build an index."""

    index_path: str
    nprobe: int | None = 8
    serve_on_gpu: bool = False
    cloner_options: dict[str, str] | None = None

    @model_validator(mode="after")
    def check_cloner_options(self) -> "InitializeIndexRequest":
        """Validate cloner options."""
        if self.serve_on_gpu and not self.cloner_options:
            raise ValueError("`cloner_options` must be provided when `serve_on_gpu` is `True`")
        return self


class InitializeIndexResponse(BaseModel):
    """Response to the initialization request."""

    success: bool
    error: str | None


class SearchRequest(BaseModel):
    """Query to search an index."""

    vectors: list[list[float]] = Field(..., description="A batch of vectors.")
    top_k: int | None = 3


class FastSearchRequest(BaseModel):
    """This is the same as SearchFaissQuery, but with the vectors serialized."""

    vectors: str = Field(..., description="A batch of serialized vectors")
    top_k: int | None = 3


class SearchResponse(BaseModel):
    """Response to the search request."""

    scores: list[list[float]] = Field(..., description="A batch of scores..")
    indices: list[list[int]] = Field(..., description="A batch of indices.")


class FastSearchResponse(BaseModel):
    """This is the same as SearchResponse, but with the vectors serialized."""

    scores: str
    indices: str
