from __future__ import annotations

from vod_search import qdrant_local_search
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
    vectors=database_vectors, port=8888, index_specification={"index": "HNSW16", "distance": "COSINE"}
) as master:
    client = master.get_client()
    print(client.search(vector=query_vectors, top_k=5))
