import argparse
import pathlib
import re
import sys
import os

src_path = pathlib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.insert(0, src_path.as_posix())  # hack to allow imports from src

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import stackprinter  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from loguru import logger  # noqa: E402
from vod_search import io  # noqa: E402
from vod_search.faiss_search import SearchFaissQuery  # noqa: E402
from vod_search.faiss_search.models import (  # noqa: E402
    FaissSearchResponse,
    FastFaissSearchResponse,
    FastSearchFaissQuery,
    InitializeIndexRequest,
    InitResponse,
)
from vod_tools.misc.exceptions import dump_exceptions_to_file  # noqa: E402

from src import vod_configs  # noqa: E402

app = FastAPI()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7678)
    parser.add_argument("--logging-level", type=str, default="INFO")
    parser.add_argument("--serve-on-gpu", action="store_true", default=False)
    return parser.parse_args()


@app.post("/load_index")
async def init_index() -> str:
    global faiss_index
    """Initialize the index."""
    logger.info("Initializing index")
    logger.info(f"Does index exists? {os.path.exists(args.index_path)}")
    logger.info(f"It's name? {args.index_path}")
    file_list = os.listdir("faiss_index/")
    logger.info(f"This exists: {file_list}")
    logger.info(f"f Does index exists? {os.path.exists(args.index_path)}")

    faiss_index = faiss.read_index(args.index_path)
    faiss_index.nprobe = args.nprobe
    logger.info(f"index initialized succesfully with {faiss_index.nprobe}")
    return "OK"


@app.get("/")
def health_check() -> str:
    """Check if the index is ready."""
    if faiss_index.ntotal == 0:
        return "ERROR: Index is empty"
    if not faiss_index.is_trained:
        return "ERROR: Index is not trained"

    return "OK"


@app.post("/search")
async def search(query: SearchFaissQuery) -> FaissSearchResponse:
    """Search the index."""
    query_vec = np.asarray(query.vectors, dtype=np.float32)
    scores, indices = faiss_index.search(query_vec, k=query.top_k)  # type: ignore
    return FaissSearchResponse(scores=scores.tolist(), indices=indices.tolist())


@app.post("/update")
async def update(r: InitializeIndexRequest) -> str:
    global faiss_index
    """Initialize the index."""
    faiss_index = faiss.read_index(r.index_path)
    # index.nprobe = r.nprobe
    # self.f = index
    # if r.serve_on_gpu:
    #     self.f = faiss.index_cpu_to_all_gpus(self.f, co=r.cloner_options)
    return "OK"


@dump_exceptions_to_file
@app.post("/fast-search")
async def fast_search(query: FastSearchFaissQuery) -> FastFaissSearchResponse:
    """Search the index."""
    try:
        query_vec = io.deserialize_np_array(query.vectors)
        if len(query_vec.shape) != 2:  # noqa: PLR2004
            raise ValueError(f"Expected 2D array, got {len(query_vec.shape)}D array")
        scores, indices = faiss_index.search(query_vec, k=query.top_k)  # type: ignore
        return FastFaissSearchResponse(
            scores=io.serialize_np_array(scores),
            indices=io.serialize_np_array(indices),
        )
    except Exception as exc:
        trace = stackprinter.format()
        raise HTTPException(status_code=500, detail=str(trace)) from exc


def run_faiss_server(host: str, port: int) -> None:
    """Start the API."""
    pattern = re.compile(r"^(http|https)://")
    host = re.sub(pattern, "", host)
    uvicorn.run(
        app, host=host, port=port, workers=1, log_level=args.logging_level.lower()
    )


def measure_memory(original_function):
    def wrapper():
        # Code to run before calling the original function
        print("Something is happening before the function is called.")
        original_function()  # Call the original function
        # Code to run after calling the original function
        print("Something is happening after the function is called.")

    return wrapper  # Return the wrapper function


args = parse_args()
logger.info("Starting API")
faiss_index = faiss.Index  # dummy object
# faiss_index = init_index(args) # build index from client command instead
run_faiss_server(args.host, args.port)
