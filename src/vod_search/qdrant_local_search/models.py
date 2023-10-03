from __future__ import annotations

from typing import Optional

import pydantic
from vod_search import rdtypes


class Response(pydantic.BaseModel):
    scores: list[list[float]] = pydantic.Field(..., description="A batch of scores")
    indices: list[list[int]] = pydantic.Field(..., description="A batch of indices")


class Query(pydantic.BaseModel):
    vectors: list[list[float]] = pydantic.Field(..., description="query vectors")
    top_k: int = pydantic.Field(default=3, description="top k similar vectors")
