from __future__ import annotations

import tempfile
import time

import faiss
import numpy as np
import rich
from loguru import logger
from rich.progress import track
from vod_search import faiss_search
from vod_tools import arguantic

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
        IVF(n_partition=100, n_probe=1),
        # HNSW(M=5, ef_construction=10, ef_search=5),
        # HNSW(M=10, ef_construction=10, ef_search=5),
    ]
    metrics = [
        "DOT",
        "L2",
    ]

    _all_index_param = []

    for preprocessing in preprocessings:
        for index_type in index_types:
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
    with faiss_search.FaissMaster(vectors, index_parameters) as master:
        client = master.get_client()
        rich.print(client)

        results = client.search(
            vector=query_vectors,
            top_k=3,
        )
        rich.print({"search_results": results})
