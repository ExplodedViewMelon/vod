import pydantic


class IndexSpecification(pydantic.BaseModel):
    index: str = pydantic.Field(..., description="Index type e.g. HNSW")
    m: int = pydantic.Field(...)
    distance: str = pydantic.Field(...)
    scalar_quantization: float = pydantic.Field(...)
    # *,
    vectors_path: str = pydantic.Field(default="", description="A path pointing to saved vectors.")
