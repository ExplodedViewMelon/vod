from __future__ import annotations

from vod_search import milvus_search
import numpy as np

vector_size: int = 128
database_vectors = np.random.random(size=(10_000, vector_size))
query_vectors = np.random.random(size=(10, vector_size))

with milvus_search.MilvusSearchMaster(vectors=database_vectors) as master:
    client = master.get_client()
    print(client.search(vector=query_vectors, top_k=5))
