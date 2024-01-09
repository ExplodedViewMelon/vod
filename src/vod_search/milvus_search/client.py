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
from rich.progress import track


# from src.vod_search.milvus_search.models import Query, Response
"""
TODO
DONE - allow for connecting to existing server
DONE - implement batch loading
implement pydantic database param specification
    - pick a few supported index types hnsw etc.
    - write down the parameters for each supported index type
    - 
implement groups / subsets, filtering etc.
implement simple benchmarking setup
clean up the connect statements. Why do these exist in the clients?
"""


class MilvusSearchClient(base.SearchClient):
    requires_vectors: bool = True

    def __init__(
        self,
        host: str,
        port: int,
        search_params: None = None,  # TODO
        collection: Collection | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._connect()
        self.collection = collection

    @property
    def _index_name(self) -> str:
        """ "Return the name of the index"""
        if self.collection:
            return self.collection.name
        else:
            return "NO_INDEX"  # TODO raise error?

    def __repr__(self) -> str:
        """Return a string representation of itself"""
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
        """Connects this python instance to the server"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            return True
        except:
            return False

    def _get_collection(self) -> None:
        """Return collection instance"""
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
        vector: Optional[ndarray],
        top_k: int = 3,
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 10},
        },
    ) -> vt.RetrievalBatch:
        if self.collection == None:
            print("No collection - getting from server")
            self._get_collection()

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

    _timeout: float = 30 * 60  # extended timeout to allow for downloading milvus docker image
    _allow_existing_server: bool = True

    def __init__(
        self,
        vectors: ndarray,  # rewrite to sequence
        *,
        host: str = "localhost",
        index_params: Optional[dict[str, Any]] = {  # TODO these must be pydantic things. One for each type of index.
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
    ) -> None:
        self.vectors = vectors
        self.host = host
        self.port = 19530  # this can't be changed I believe.
        self.collection = None
        self.index_params = index_params
        self.batch_size = 1000

        skip_setup = False
        super().__init__(skip_setup)

    def _on_init(self) -> None:
        """Connect to server and build index"""
        connections.connect("default", host=self.host, port=self.port)
        self._delete_existing_collection()
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

    def _delete_existing_collection(self) -> None:
        if utility.has_collection("index_name"):
            logger.info("Collection already exists, deleting.")
            utility.drop_collection("index_name")

    def _build_index(self) -> None:
        logger.info("Creating collection 'index_name'")
        if len(self.vectors.shape) != 2:
            raise ValueError(f"Expected a NxD vectors, got {self.vectors.shape}")
        N, D = self.vectors.shape

        # TODO fill make it possible to specify all the below in some passed struct
        # or maybe not idk. but rewrite such that it support groups / subsets for filtering etc.
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=D),
        ]
        schema = CollectionSchema(fields, "Milvus database - so far so good")
        collection = Collection("index_name", schema, consistency_level="Strong")

        for j in track(range(0, N, self.batch_size), description=f"Milvus: Ingesting {N} vectors of size {D}"):
            entities = [
                list(range(j, j + self.batch_size)),
                self.vectors[j : j + self.batch_size],
            ]  # rewrite to type sequence
            collection.insert(entities)

        collection.flush()  # seals the unfilled buckets
        collection.create_index("embeddings", self.index_params)
        collection.load()  # load index into server
        self.collection = collection
