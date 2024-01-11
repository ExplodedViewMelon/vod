from __future__ import annotations

from vod_search import milvus_search
import numpy as np
from vod_search.models import *


def all_index_param():
    preprocessings = [
        None,
        ProductQuantization(m=4),
        ProductQuantization(m=8),
        ScalarQuantization(n=4),
        ScalarQuantization(n=8),
    ]
    index_types = [
        IVF(n_partition=10, n_probe=1),
        IVF(n_partition=100, n_probe=10),
        HNSW(M=5, ef_construction=10, ef_search=15),
        HNSW(M=10, ef_construction=100, ef_search=150),
    ]
    metrics = [  # add cosine
        "L2",
        "IP",
        "COSINE",
    ]

    _all_index_param = []

    for index_type in index_types:
        for preprocessing in preprocessings:
            for metric in metrics:
                _all_index_param.append(
                    IndexParameters(
                        preprocessing=preprocessing,
                        index_type=index_type,
                        metric=metric,
                        top_k=10,
                    )
                )
    return _all_index_param


# specify test sizes
dataset_size: int = 10_000
vector_size: int = 128

# make dummy data
database_vectors = np.random.random(size=(dataset_size, vector_size))
vectors = np.random.random(size=(dataset_size, vector_size)).astype("float32")
query_vectors = np.random.random(size=(10, vector_size))

# index_parameters = IndexParameters(
#     # preprocessing=ProductQuantization(m=4),
#     # index_type=IVF(n_partition=100, n_probe=10),
#     preprocessing=None,
#     index_type=HNSW(ef_construction=10, ef_search=10, M=5),
#     metric="L2",
#     top_k=5,
# )

for index_parameters in all_index_param():
    print("Testing with", index_parameters)
    # Spin up a Faiss server
    with milvus_search.MilvusSearchMaster(vectors, index_parameters) as master:  # type: ignore
        client = master.get_client()
        print(client.search(vector=query_vectors))
