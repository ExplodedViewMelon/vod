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
from vod_search.models import IndexSpecification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int)
    return parser.parse_args()


class QdrantDatabase:
    """init simple qdrant database with a single collection"""

    def __init__(
        self,
    ) -> None:
        self.collection_name = "QdrantLocalDatabase"
        self.db = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

    def _search_single(self, query_vector: list[float], limit: int = 3):
        return self.db.search(collection_name=self.collection_name, query_vector=query_vector, limit=limit)

    def search(self, query_vectors: list[list[float]], top_k: int) -> Response:
        """Search database for top_k similar vectors for batch"""
        search_queries = [models.SearchRequest(vector=vector, limit=top_k) for vector in query_vectors]
        results = self.db.search_batch(collection_name=self.collection_name, requests=search_queries)

        # shape result into Response format
        scores: list[list[float]] = []
        indices: list[list[int]] = []
        for query in results:
            scores.append([val.score for val in query])
            indices.append([int(val.id) for val in query])
        return Response(scores=scores, indices=indices)

    def build(self, vector_path: str, index_specification: IndexSpecification):
        # load vectors
        vectors: np.ndarray = np.load(vector_path, allow_pickle=True)
        vectors_n, vectors_dim = vectors.shape

        # init database
        self.db = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")
        self.collection_name = "QdrantLocalDatabase"

        # get distance object
        distance = {
            "COSINE": models.Distance.COSINE,
            "DOT": models.Distance.DOT,
            "EUCLID": models.Distance.EUCLID,
            "L2": models.Distance.EUCLID,
            "COSINE": models.Distance.COSINE,
        }[index_specification.distance]

        # create index
        self.db.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vectors_dim, distance=distance),
            hnsw_config=models.HnswConfigDiff(
                m=index_specification.m,
                ef_construct=100,  # TODO specify somewhere. can also be specified during search
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=index_specification.scalar_quantization,
                    always_ram=True,
                ),
            ),
        )
        # quantization_config=models.ProductQuantization(
        #     product=models.ProductQuantizationConfig(
        #         compression=models.CompressionRatio.X16,
        #         always_ram=True,
        #     ),
        # )

        # add vectors
        ids: list[ExtendedPointId] = list(range(len(vectors)))
        self.db.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors.tolist(),
                payloads=None,
            ),
        )


args = parse_args()  # parse host and port
DB = QdrantDatabase()  # run qdrant
app = FastAPI()  # run server


@app.get("/")
def health_check() -> str:
    """Check if the server is running."""
    return "OK"


@app.post("/build")
async def build(index_specification: IndexSpecification) -> None:
    vectors_path = index_specification.vectors_path
    print("recived build command with", vectors_path, index_specification)
    DB.build(vectors_path, index_specification)


@app.post("/search")
async def search(query: Query) -> Response:
    """returns response from search query"""
    return DB.search(query.vectors, top_k=query.top_k)


def run_qdrant_local_server(host: str, port: int) -> None:
    """Start the API and server."""
    pattern = re.compile(r"^(http|https)://")
    host = re.sub(pattern, "", host)
    uvicorn.run(app, host=host, port=port, workers=1)


def _test():
    # a small local test for checking database search
    print(asyncio.run(search(Query(vectors=np.random.random((10, args.vector_size)).tolist()))))


if __name__ == "__main__":
    # test()
    run_qdrant_local_server(args.host, args.port)
