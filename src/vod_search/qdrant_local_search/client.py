from __future__ import annotations

from typing import Optional, Any
import pydantic
from vod_search import base
import vod_types as vt
import abc
import sys
from pathlib import Path
import requests
from copy import copy
import os
import numpy as np
from numpy import ndarray
import torch
from vod_search.qdrant_local_search.models import Query, Response
from vod_search.models import IndexSpecification


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

    def save_vectors_as_file(self, vectors: ndarray) -> str:
        # since local qdrant cannot save or read indexes, vectors are passed to server using this file.
        np.save("qdrant_local_vectors.npy", vectors)
        return "qdrant_local_vectors.npy"

    def build(self, vectors: ndarray, index_specification: IndexSpecification, timeout=60 * 10):
        # save vectors in shared folder
        vectors_path = self.save_vectors_as_file(vectors)
        index_specification.vectors_path = vectors_path  # add to index_specification

        # send build command to server
        response = requests.post(f"{self.url}/build", json=index_specification.model_dump(), timeout=timeout)
        response.raise_for_status()

    def search(
        self,
        *,
        vector: ndarray,
        top_k: int,
        timeout: int = 120,
        # group: list[str | int] | None = None,
        # section_ids: list[list[str | int]] | None = None,
    ) -> vt.RetrievalBatch:
        query = Query(vectors=vector.tolist(), top_k=top_k)
        response = requests.post(
            f"{self.url}/search",
            json=query.model_dump(),
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
        return vt.RetrievalBatch(indices=indices, scores=scores)


class QdrantLocalSearchMaster(base.SearchMaster[QdrantLocalSearchClient], abc.ABC):
    # maybe delete vectors file when closing object?
    # move creation of .npy out of the master object. Input path instead.
    # or maybe not. Faiss gets a file, qdrant gets the vectors...
    def __init__(
        self,
        port=6333,
    ) -> None:
        self.host = "http://localhost"
        self.port = port
        super().__init__()

    def __repr__(self):
        return "QdrantLocalSearchMaster"

    def get_client(self) -> QdrantLocalSearchClient:
        return QdrantLocalSearchClient(self.host, self.port)

    def _make_cmd(self) -> list[str]:
        # get the path to the server script
        # add arguments to server.py
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return super().__exit__(exc_type, exc_val, exc_tb)
