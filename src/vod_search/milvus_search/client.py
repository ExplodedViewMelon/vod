from __future__ import annotations
from typing import Any, Optional

import json
import pydantic
from vod_search import base, rdtypes
import abc
import sys
from pathlib import Path
import requests
from copy import copy
import os
import numpy as np
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

# from src.vod_search.milvus_search.models import Query, Response


class MilvusSearchClient(base.SearchClient):
    def __init__(self, collection: Collection, host: str, port: int) -> None:
        self.collection = collection  # TODO check that this is ok
        self.host = host
        self.port = port
        # connections.connect("default", host=self.host, port=self.port)

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    def ping(self) -> bool:
        """Ping the server."""
        # TODO implement
        return True

    def search(
        self,
        *,
        vector: np.ndarray,
        group: list[str | int] | None = None,
        section_ids: list[list[str | int]] | None = None,
        top_k: int = 3,
    ) -> rdtypes.RetrievalBatch[rdtypes.Ts]:  # TODO specify return value
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result: SearchResult = self.collection.search(vector.tolist(), "embeddings", search_params, limit=top_k, _async=False)  # type: ignore
        scores = np.asarray([hits.distances for hits in result])
        indices = np.asarray([hits.ids for hits in result])

        return rdtypes.RetrievalBatch(scores, indices)  # type: ignore # TODO update indexing for batches of queries


class MilvusSearchMaster(base.SearchMaster[MilvusSearchClient], abc.ABC):
    def __init__(self, vectors: np.ndarray, skip_setup: bool = False) -> None:
        self._allow_existing_server = True
        self.vectors = vectors
        self.host = "localhost"
        self.port = 19530
        super().__init__(skip_setup)
        self.collection = self._build_index()

    def _build_index(self) -> Collection:
        connections.connect("default", host=self.host, port=self.port)
        if utility.has_collection("database_oas"):
            logger.info("Collection already exists, deleting.")
            utility.drop_collection("database_oas")

        logger.info("Creating collection 'database_oas'")
        database_size, vector_size = self.vectors.shape
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        ]
        schema = CollectionSchema(fields, "Milvus database - so far so good")
        collection = Collection("database_oas", schema, consistency_level="Strong")
        entities = [list(range(database_size)), self.vectors]
        insert_result = collection.insert(entities)
        collection.flush()  # seals the unfilled buckets
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index("embeddings", index)
        collection.load()  # load index into server
        return collection

    def _on_init(self) -> None:
        # self._build_index()
        return

    def _on_exit(self) -> None:
        utility.drop_collection("database_oas")

    def get_client(self) -> MilvusSearchClient:
        return MilvusSearchClient(collection=self.collection, host=self.host, port=self.port)

    def _make_cmd(self) -> list[str]:
        # TODO run docker server
        return super()._make_cmd()
