from __future__ import annotations

from vod_search import milvus_search
import numpy as np
from vod_search.models import *

# specify test sizes
dataset_size: int = 10_000
vector_size: int = 128

# make dummy data
database_vectors = np.random.random(size=(dataset_size, vector_size))
vectors = np.random.random(size=(dataset_size, vector_size)).astype("float32")
query_vectors = np.random.random(size=(10, vector_size))

# make parameter specification
index_parameters = IndexParameters(
    # preprocessing=ProductQuantization(m=4),
    # index_type=IVF(n_partition=100, n_probe=10),
    preprocessing=None,
    index_type=HNSW(ef_construction=10, ef_search=10, M=5),
    metric="L2",
    top_k=5,
)

with milvus_search.MilvusSearchMaster(vectors, index_parameters) as master:  # type: ignore
    client = master.get_client()
    print(client.search(vector=query_vectors, top_k=5))
