from __future__ import annotations

from vod_search import qdrant_local_search
from vod_search.models import IndexSpecification
import numpy as np

""" TODO
Parameters for testing:
size of data, type of data? both query and database
PQ of vectors
indexing type
distance metric
aux data, filtering etc.

Metrics for evaluation:
Speed, both mean and 99 percentile
Memory <- how in the world am I going to do that? Rely on docker measurements?
Recall and recall@k (number of queries that has the ground truth nearest neighbour in it's k returned results.)

Streamline parameters.

There is an infeasable number of combinations for testing. I ought to either:
Set up a bunch of conditions, refinement cannot be harsher than preprocessing e.g.,
or talk to Andreas about a number of tests.
This is an argument for making a streamlined index factory. Then you can test these things runningly.

Write Valentin about this matter.

"""

vector_size: int = 128
database_vectors = np.random.random(size=(10_000, vector_size))
query_vectors = np.random.random(size=(10, vector_size))

with qdrant_local_search.QdrantLocalSearchMaster(
    port=8888,
) as master:
    client = master.get_client()
    client.build(
        database_vectors,
        IndexSpecification(
            index="HNSW", m=32, distance="COSINE", scalar_quantization=0.99, vectors_path="qdrant_local_vectors.npy"
        ),
    )
    print(client.search(vector=query_vectors, top_k=5))
