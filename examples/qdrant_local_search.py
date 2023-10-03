from __future__ import annotations

from vod_search import qdrant_local_search
import numpy as np

vec_size: int = 128
dummy_vectors = np.random.random((1000, vec_size))

query_vectors = np.random.random(size=(10, vec_size))

with qdrant_local_search.QdrantLocalSearchMaster("dummy/path/") as master:
    client = master.get_client()
    print(client.search(vector=query_vectors))
