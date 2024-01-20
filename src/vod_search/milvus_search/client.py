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

from vod_search.models import *


class MilvusSearchClient(base.SearchClient):
    requires_vectors: bool = True

    def __init__(
        self,
        host: str,
        port: int,
        master: MilvusSearchMaster,
        collection: Collection | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._connect()
        self.master = master
        self.collection = collection

    @property
    def _index_name(self) -> str:
        """ "Return the name of the index"""
        if self.collection:
            return self.collection.name
        else:
            return "NO_INDEX"  # NOTE should this raise an error?

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

    def _connect(self) -> bool:  # NOTE this also serves as a ping
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

    def _make_search_params(self):
        if isinstance(self.master.index_parameters.index_type, IVF):
            return {
                "metric_type": self.master.index_parameters.metric,
                "params": {
                    "nprobe": self.master.index_parameters.index_type.n_probe,
                },
            }
        elif isinstance(self.master.index_parameters.index_type, HNSW):
            return {
                "metric_type": self.master.index_parameters.metric,
                "params": {
                    "ef": self.master.index_parameters.index_type.ef_search,
                },
            }

    def search(
        self,
        *,
        vector: Optional[ndarray],
        top_k: int,
    ) -> vt.RetrievalBatch:
        if self.collection == None:
            print("No collection - getting from server")
            self._get_collection()

        # top_k: int = self.master.index_parameters.top_k
        result: SearchResult = self.collection.search(vector.tolist(), "embeddings", param=self._make_search_params(), limit=top_k, _async=False)  # type: ignore
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
        index_parameters: IndexParameters,
        *,
        host: str = "localhost",
    ) -> None:
        self.vectors = vectors
        self.host = host
        self.port = 19530  # this can't be changed I believe.
        self.collection = None
        self.index_parameters = index_parameters
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
        return MilvusSearchClient(host=self.host, port=self.port, master=self, collection=self.collection)

    def _make_cmd(self) -> list[str]:
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

    def _make_index_parameters(self):
        preprocessing: None | ProductQuantization | ScalarQuantization = self.index_parameters.preprocessing
        index_type: HNSW | IVF = self.index_parameters.index_type

        if isinstance(index_type, IVF):
            if isinstance(preprocessing, ProductQuantization):
                return {
                    "index_type": "IVF_PQ",
                    "metric_type": self.index_parameters.metric,
                    "params": {
                        "nlist": index_type.n_partition,
                        "m": preprocessing.m,
                        # "nbits": preprocessing.n_bits,
                    },
                }
            elif isinstance(preprocessing, ScalarQuantization):
                assert preprocessing.n == 8, "SQ8 is the only supported"
                return {
                    "index_type": "IVF_SQ8",
                    "metric_type": self.index_parameters.metric,
                    "params": {
                        "nlist": index_type.n_partition,
                    },
                }
            else:  # preprocessing is None
                return {
                    "index_type": "IVF_FLAT",
                    "metric_type": self.index_parameters.metric,
                    "params": {
                        "nlist": index_type.n_partition,
                    },
                }

        if isinstance(index_type, HNSW):
            assert preprocessing == None, "Milvus does not support quantizers for hnsw."
            return {
                "index_type": "HNSW",
                "metric_type": self.index_parameters.metric,
                "params": {
                    "M": index_type.M,
                    "efConstruction": index_type.ef_construction,
                },
            }

    def _build_index(self) -> None:
        logger.info("Creating collection 'index_name'")
        if len(self.vectors.shape) != 2:
            raise ValueError(f"Expected a NxD vectors, got {self.vectors.shape}")
        N, D = self.vectors.shape

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=D),
        ]
        schema = CollectionSchema(fields, "Milvus database - so far so good")
        collection = Collection("index_name", schema, consistency_level="Strong")

        for j in track(range(0, N, self.batch_size), description=f"Milvus: Ingesting {N} vectors of size {D}"):
            to_insert = self.vectors[j : j + self.batch_size]
            arbitrary_index = list(range(j, j + len(to_insert)))
            entities = [
                arbitrary_index,
                to_insert,
            ]  # rewrite to type sequence
            collection.insert(entities)

        collection.flush()  # seals the unfilled buckets
        index_parameters = self._make_index_parameters()
        collection.create_index("embeddings", index_parameters)
        collection.load()  # load index into server
        self.collection = collection

    def __repr__(self) -> str:
        return f"index: milvus {self.index_parameters}"
