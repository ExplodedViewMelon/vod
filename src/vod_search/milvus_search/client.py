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
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        super().__init__()

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

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
        # TODO preprocess and send query to server
        # TODO get result, process and return

        return super().search(text=text, vector=vector, group=group, section_ids=section_ids, top_k=top_k)


class MilvusSearchMaster(base.SearchMaster[MilvusSearchClient], abc.ABC):
    def __init__(self, vectors: np.ndarray, skip_setup: bool = False) -> None:
        self.vectors = vectors
        self.host = "http://localhost"
        self.port = 9999
        super().__init__(skip_setup)

    def get_client(self) -> MilvusSearchClient:
        return MilvusSearchClient(self.host, self.port)

    def _make_cmd(self) -> list[str]:
        # TODO run docker server
        return super()._make_cmd()
