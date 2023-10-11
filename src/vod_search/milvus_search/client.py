from __future__ import annotations
from typing import Any, Optional

import pydantic
from vod_search import base, rdtypes
import abc
import sys
from pathlib import Path
import requests
from copy import copy
import os
import numpy as np

# from src.vod_search.milvus_search.models import Query, Response


class MilvusSearchClient(base.SearchClient):
    def __init__(self) -> None:
        super().__init__()

    def url(self) -> str:
        return ""

    def ping(self) -> bool:
        return super().ping()

    def search(
        self,
        *,
        text: list[str],
        vector: rdtypes.Ts | None = None,
        group: list[str | int] | None = None,
        section_ids: list[list[str | int]] | None = None,
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[rdtypes.Ts]:
        return super().search(text=text, vector=vector, group=group, section_ids=section_ids, top_k=top_k)


class MilvusSearchMaster(base.SearchMaster[MilvusSearchClient], abc.ABC):
    def __init__(self, skip_setup: bool = False) -> None:
        super().__init__(skip_setup)

    def get_client(self) -> MilvusSearchClient:
        return super().get_client()

    def _make_cmd(self) -> list[str]:
        return super()._make_cmd()

    def _make_env(self) -> dict[str, Any]:
        return super()._make_env()
