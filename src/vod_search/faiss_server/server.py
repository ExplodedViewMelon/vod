import io
from typing import Type

import faiss
import fastapi
import numpy as np
from fastapi import APIRouter, HTTPException, Response, status
from vod_search.io import base64
from .models import (
    FastSearchRequest,
    FastSearchResponse,
    InitializeIndexRequest,
    SearchRequest,
    SearchResponse,
)

BATCH_DIMENSION = 2


class FaissServer:
    """Faiss server class."""

    f: faiss.Index

    def __init__(self) -> None:
        self.router = APIRouter()
        self.router.add_api_route("/", self.root, methods=["GET"])
        self.router.add_api_route("/initialize", self.root, methods=["POST"])
        self.router.add_api_route("/search", self.search, methods=["POST"])
        self.router.add_api_route("/fast-search", self.fast_search, methods=["POST"])

    @staticmethod
    def _deserialize_np_array(encoded_array: str, *, dtype: None | Type[np.dtype] = None) -> np.ndarray:
        """Deserializes a numpy array from a string."""
        np_bytes = base64.urlsafe_b64decode(encoded_array)
        load_bytes = io.BytesIO(np_bytes)
        loaded_np = np.load(load_bytes, allow_pickle=True)
        if dtype is not None:
            loaded_np = loaded_np.astype(dtype)
        return loaded_np

    @staticmethod
    def _serialize_np_array(array: np.ndarray) -> str:
        """Serializes a numpy array to a string."""
        bytes_buffer = io.BytesIO()
        np.save(bytes_buffer, array, allow_pickle=True)
        bytes_buffer = bytes_buffer.getvalue()
        return base64.urlsafe_b64encode(bytes_buffer).decode("utf-8")

    def root(self) -> Response:
        """Check if the index is ready."""
        if not self.f:
            raise HTTPException(status_code=409, detail="index doesn't exist")
        if self.f.ntotal == 0:
            raise HTTPException(status_code=409, detail="index is empty")
        if not self.f.is_trained:
            raise HTTPException(status_code=409, detail="index is not trained")

        return Response(content="OK", status_code=fastapi.status.HTTP_200_OK)

    def initialize(self, r: InitializeIndexRequest) -> Response:
        """Initialize the index."""
        index = faiss.read_index(r.index_path)
        index.nprobe = r.nprobe
        self.f = index
        if r.serve_on_gpu:
            self.f = faiss.index_cpu_to_all_gpus(self.f, co=r.cloner_options)
        return Response(content="OK", status_code=status.HTTP_200_OK)

    def search(self, r: SearchRequest) -> SearchResponse:
        """Search the index."""
        query_vec = np.asarray(r.vectors, dtype=np.float32)
        scores, indices = self.f.search(query_vec, k=r.top_k)  # type: ignore
        return SearchResponse(scores=scores.tolist(), indices=indices.tolist())

    def fast_search(self, r: FastSearchRequest) -> FastSearchResponse:
        """Search the index."""
        try:
            query_vec = self._deserialize_np_array(r.vectors)
            if len(query_vec.shape) != BATCH_DIMENSION:
                raise ValueError(f"Expected {BATCH_DIMENSION}D array, got {len(query_vec.shape)}D array")
            scores, indices = self.f.search(query_vec, k=r.top_k)  # type: ignore
            return FastSearchResponse(
                scores=self._serialize_np_array(scores),
                indices=self._serialize_np_array(indices),
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
