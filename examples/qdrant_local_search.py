from __future__ import annotations

from vod_search import qdrant_local_search
import numpy as np

""" TODO
Figure how to build database first or at least specify how many datapoints to test on.
Maybe make an API call to build a dummy database
"""

vec_size: int = 128
query_vectors = np.random.random(size=(10, vec_size))

with qdrant_local_search.QdrantLocalSearchMaster() as master:
    client = master.get_client()
    print(client.search(vector=query_vectors, top_k=5))
