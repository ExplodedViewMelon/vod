from __future__ import annotations
from typing import Any, Optional

import json
import numba
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
from pymilvus import (
    SearchResult,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from loguru import logger
from typing import Any, Iterable, Optional


# from src.vod_search.milvus_search.models import Query, Response


class MilvusSearchClient(base.SearchClient):
    requires_vectors: bool = True

    def __init__(
        self,
        host: str,
        port: int,
        grpc_port: None | int = None,  # TODO
        search_params: None = None,  # TODO
        supports_groups: bool = True,  # TODO
        collection: Collection | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._connect()
        self.collection = collection

    @property
    def _index_name(self) -> str:
        if self.collection:
            return self.collection.name
        else:
            return "NO_INDEX"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}[{self.host}:{self.port}]("
            f"requires_vectors={self.requires_vectors}, "
            f"index_name={self._index_name})"
        )

    def size(self) -> int:
        """Return the number of vectors in the index."""
        if self.collection:
            return self.collection.num_entities
        else:
            return 0

    def _connect(self) -> bool:
        try:
            connections.connect("default", host=self.host, port=self.port)
            return True
        except:
            return False

    def _get_collection(self) -> None:
        self.collection = Collection("index_name")

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    def ping(self) -> bool:
        """Ping the server."""
        return self._connect()

    def search(
        self,
        *,
        text: Optional[list[str]] = None,  # noqa: ARG002
        vector: Optional[ndarray],
        group: Optional[list[str | int]] = None,
        section_ids: Optional[list[list[str | int]]] = None,  # noqa: ARG002
        top_k: int = 3,
    ) -> vt.RetrievalBatch:
        if self.collection == None:
            print("No collection passed - getting from server")
            self._get_collection()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result: SearchResult = self.collection.search(vector.tolist(), "embeddings", search_params, limit=top_k, _async=False)  # type: ignore

        return _search_batch_to_rdtypes(result, top_k)


@numba.jit(forceobj=True, looplift=True)
def _search_batch_to_rdtypes(batch: SearchResult, top_k: int) -> vt.RetrievalBatch:
    """Convert a batch of search results to rdtypes."""
    scores = np.full((len(batch), top_k), -np.inf, dtype=np.float32)
    indices = np.full((len(batch), top_k), -1, dtype=np.int64)
    max_j = -1
    for i, row in enumerate(batch):
        for j, p in enumerate(row):
            scores[i, j] = p.score
            indices[i, j] = p.id
            if j > max_j:
                max_j = j

    return vt.RetrievalBatch(
        scores=scores[:, : max_j + 1],
        indices=indices[:, : max_j + 1],
    )


class MilvusSearchMaster(base.SearchMaster[MilvusSearchClient], abc.ABC):
    """A class that manages a search server."""

    _allow_existing_server: bool = True

    def __init__(
        self,
        vectors: ndarray,
        *,
        groups: Optional[Iterable[str | int]] = None,
        host: str = "localhost",
        grpc_port: None | int = 6334,
        index_name: Optional[str] = "database",
        persistent: bool = True,
        exist_ok: bool = True,
        skip_setup: bool = False,
        index_body: Optional[dict[str, Any]] = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
        search_params: Optional[dict[str, Any]] = None,
        force_single_collection: bool = True,
    ) -> None:
        self._allow_existing_server = True
        self.vectors = vectors
        self.host = host
        self.port = 19530  # this can't be changed I believe.
        self.collection = None
        self.index_name = index_name
        self.index_body = index_body
        super().__init__(skip_setup)

    def _build_index(self) -> None:
        connections.connect("default", host=self.host, port=self.port)
        if utility.has_collection("index_name"):
            logger.info("Collection already exists, deleting.")
            utility.drop_collection("index_name")

        logger.info("Creating collection 'index_name'")
        database_size = len(self.vectors)
        vector_shape = self.vectors[0].shape
        vector_size = vector_shape[0]
        if len(vector_shape) != 1:
            raise ValueError(f"Expected a 1D vectors, got {vector_shape}")

        # TODO fill make it possible to specify all the below in some passed struct
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        ]
        schema = CollectionSchema(fields, "Milvus database - so far so good")
        collection = Collection("index_name", schema, consistency_level="Strong")
        entities = [list(range(database_size)), self.vectors]
        insert_result = collection.insert(entities)
        collection.flush()  # seals the unfilled buckets
        collection.create_index("embeddings", self.index_body)
        collection.load()  # load index into server
        self.collection = collection

    def _on_init(self) -> None:
        self._build_index()

    def _on_exit(self) -> None:
        utility.drop_collection("index_name")

    def get_client(self) -> MilvusSearchClient:
        return MilvusSearchClient(host=self.host, port=self.port, collection=self.collection)

    def _make_cmd(self) -> list[str]:
        # TODO # docker compose up -d
        return [
            "docker",
            "compose",
            "up",
            "-d",
        ]
