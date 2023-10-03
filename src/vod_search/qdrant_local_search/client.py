from __future__ import annotations

from typing import Optional
import pydantic
from vod_search import base, rdtypes
import abc
import sys
from pathlib import Path
import requests
from copy import copy
import os
import numpy as np
import torch
from src.vod_search.qdrant_local_search.models import Query, Response

# TODO
# implement functions for searching locally with qdrant
# create server inteface
# create master / client thing


class QdrantLocalSearchClient(base.SearchClient):
    requires_vectors = True

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    def ping(self, timeout: float = 120) -> bool:
        """Ping the server."""
        try:
            response = requests.get(f"{self.url}/", timeout=timeout)
            return True
        except requests.exceptions.ConnectionError:
            return False

        # response.raise_for_status()
        # return "OK" in response.text

    def search(
        self,
        *,
        vector: rdtypes.Ts,
        timeout: int = 120,
        # group: list[str | int] | None = None,
        # section_ids: list[list[str | int]] | None = None,
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[rdtypes.Ts]:
        query = Query(vectors=vector.tolist(), top_k=top_k)
        response = requests.post(
            f"{self.url}/search",
            json=query.dict(),
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        indices_list = data["indices"]
        scores_list = data["scores"]
        # TODO understand this type casting
        input_type = type(vector)
        cast_fn = {
            torch.Tensor: torch.tensor,
            np.ndarray: np.array,
        }[input_type]
        indices = cast_fn(indices_list)
        scores = cast_fn(scores_list)
        return rdtypes.RetrievalBatch(indices=indices, scores=scores)


class QdrantLocalSearchMaster(base.SearchMaster[QdrantLocalSearchClient], abc.ABC):
    def __init__(self, skip_setup: bool = False) -> None:
        self.host = "http://localhost"
        self.port = 6333
        super().__init__(skip_setup)

    def get_client(self) -> QdrantLocalSearchClient:
        return QdrantLocalSearchClient(self.host, self.port)

    def _make_cmd(self) -> list[str]:
        # add arguments to server.py
        # building of index is done in master, not in server
        # get the path to the server script
        server_run_path = Path(__file__).parent / "server.py"
        executable_path = sys.executable
        return [
            str(executable_path),
            str(server_run_path),
            "--host",
            str(self.host),
            "--port",
            str(self.port),
        ]

    def _make_env(self) -> dict[str, str]:
        env = copy(dict(os.environ))
        if "KMP_DUPLICATE_LIB_OK" not in env:
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["LOGURU_LEVEL"] = "DEBUG"
        # add the local path, so importing the library will work.
        if "PYTHONPATH" not in env:
            env["PYTHONPATH"] = str(Path.cwd())
        else:
            os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{Path.cwd()}"
        return env
