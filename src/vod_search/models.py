import pydantic


class IndexSpecification(pydantic.BaseModel):
    index: str
    m: int
    distance: str
    scalar_quantization: float
