from fastapi import FastAPI
import uvicorn
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import pydantic
import asyncio
import argparse
from loguru import logger
from qdrant_client.http import models
from src.vod_search.qdrant_local_search.models import Response, Query
from qdrant_client.models import ExtendedPointId


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    return parser.parse_args()


class QdrantDatabase:

    """init simple qdrant database with a single collection"""

    def __init__(self, vector_size: int, name: str) -> None:
        self.db = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")
        self.name = name
        self.db.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

    def ingest_data(self, vectors: np.ndarray) -> None:
        ids: list[ExtendedPointId] = list(range(len(vectors)))
        self.db.upsert(
            collection_name=self.name,
            points=models.Batch(
                ids=ids,
                vectors=vectors.tolist(),
                payloads=None,
            ),
        )

    def _search_single(self, query_vector: list[float], limit: int = 3):
        return self.db.search(
            collection_name=self.name, query_vector=query_vector, limit=limit
        )

    def search(self, query_vectors: list[list[float]], top_k: int) -> Response:
        # get batches request
        search_queries = [
            models.SearchRequest(vector=vector, limit=top_k) for vector in query_vectors
        ]
        # execute
        results = self.db.search_batch(
            collection_name=self.name, requests=search_queries
        )

        # shape result into Response format
        scores: list[list[float]] = []
        indices: list[list[int]] = []
        for query in results:
            scores.append([val.score for val in query])
            indices.append([int(val.id) for val in query])
        return Response(scores=scores, indices=indices)


logger.info("Server waking up...")
print("Server waking up -_-")
args = parse_args()
vec_size = 128
DB = QdrantDatabase(vec_size, "test_database")

logger.info("Creating dummy database")
DB.ingest_data(np.random.random((1000, vec_size)))  # add 1000 random vectors
# DB.load_index(args.index_path)
# DB.save_index()
app = FastAPI()


@app.get("/")
def health_check() -> str:
    """Check if the server is running."""
    return "hello"


@app.post("/search")
async def search(query: Query) -> Response:
    """returns response from search query"""
    return DB.search(query.vectors, top_k=query.top_k)


def run_faiss_server(host: str, port: int) -> None:
    """Start the API and server."""
    pattern = re.compile(r"^(http|https)://")
    host = re.sub(pattern, "", host)
    uvicorn.run(app, host=host, port=port, workers=1)


def test():
    # a small test
    print(asyncio.run(search(Query(vectors=np.random.random((10, vec_size)).tolist()))))


if __name__ == "__main__":
    # test()
    print("So far so good MIAVMIAV ")
    run_faiss_server(args.host, args.port)
