from __future__ import annotations

import time

import numpy as np
import rich
from loguru import logger
from rich.progress import track
from vod_search import qdrant_search
from vod_tools import arguantic


class Args(arguantic.Arguantic):
    """Arguments for the script."""

    dataset_size: int = 3_000
    n_categories: int = 100
    vector_size: int = 128
    batch_size: int = 10
    top_k: int = 100
    n_trials: int = 10
    persistent: bool = False


def run(args: Args) -> None:
    """Run the script."""
    index_name = f"test_collection_{args.dataset_size}_{args.vector_size}_{args.n_categories}"

    # Dataset
    vectors = np.random.randn(args.dataset_size, args.vector_size).astype("float32")
    groups = np.linspace(0, args.n_categories, len(vectors)).astype("int64")

    # Spin up a server
    with qdrant_search.QdrantSearchMaster(
        vectors=vectors,  # type: ignore
        groups=groups,
        index_name=index_name,
        qdrant_body={
            "shard_number": 1,
            "hnsw_config": {
                "ef_construct": 256,
                "m": 16,
            },
            "quantization_config": {
                "product": {
                    "compression": "x8",
                    "always_ram": True,
                },
            },
        },
        search_params={
            "ef_search": 256,
        },
        persistent=args.persistent,
    ) as master:
        client = master.get_client()
        rich.print(client)
        logger.info(f"Client: {client.size()} records")

        query_vecs = np.random.randn(args.bs, args.vector_size).astype("float32")
        query_groups = np.random.randint(0, args.n_categories, size=args.bs).astype("int64")

        results = client.search(
            vector=query_vecs,
            group=query_groups,  # type: ignore
            top_k=3,
        )
        rich.print(
            {
                "search_results": results,
                "query_groups": query_groups,
                "results_groups": [[groups[i] for i in row] for row in results.indices],
            }
        )

        # Benchmark
        logger.info("Benchmarking...")
        start = time.perf_counter()
        for _ in track(range(args.n_trials), description="Benchmarking Qdrant"):
            query_vecs = np.random.randn(args.bs, args.vector_size).astype("float32")
            query_groups = np.random.randint(0, args.n_categories, size=args.bs).astype("int64")
            results = client.search(
                vector=query_vecs,
                group=query_groups,  # type: ignore
                top_k=args.top_k,
            )
        end = time.perf_counter()
        logger.info(f"Qdrant: {1000*(end - start) / args.n_trials:.3f} ms/batch")


if __name__ == "__main__":
    args = Args.parse()
    rich.print(args)
    run(args)
