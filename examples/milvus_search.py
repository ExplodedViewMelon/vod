from __future__ import annotations

from vod_search import milvus_search
import numpy as np


dataset_size: int = 10_000
vector_size: int = 128
database_vectors = np.random.random(size=(dataset_size, vector_size))
vectors = np.random.random(size=(dataset_size, vector_size)).astype("float32")

query_vectors = np.random.random(size=(10, vector_size))

with milvus_search.MilvusSearchMaster(vectors) as master:  # type: ignore
    client = master.get_client()
    print(client.search(vector=query_vectors, top_k=5))
