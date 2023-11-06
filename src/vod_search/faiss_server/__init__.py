"""Index and serve vector bases using faiss server v2."""

from .models import (
    FastSearchRequest,
    FastSearchResponse,
    InitializeIndexRequest,
    InitializeIndexResponse,
    SearchRequest,
    SearchResponse,
)
from .server import FaissServer
