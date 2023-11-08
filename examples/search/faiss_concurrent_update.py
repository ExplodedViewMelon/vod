from vod_search.faiss_search import FaissMaster, FaissClient, build_faiss_index
import numpy as np
import faiss
import tempfile
import asyncio


async def repeat_querying(client):
    while True:
        query = np.ones(shape=(1, 128))
        print("queried")
        await asyncio.sleep(1)


async def build_index(index_path: str):
    # build faiss index
    n = 1000
    d = 128
    index = build_faiss_index(np.random.random(size=(n, d)), factory_string="PCA80,Flat")
    faiss.write_index(index, index_path)
    await asyncio.sleep(10)


async def main():
    # save faiss index
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = f"{tmpdir}/index.faiss"
        await build_index(index_path)

        # spin up faiss server
        with FaissMaster(index_path) as master:
            client = master.get_client()

            for _ in range(10):
                # repeatably query the index
                query_task = asyncio.create_task(repeat_querying(client))

                # start building a new index
                build_index_task = asyncio.create_task(build_index(index_path))

                # when new index is built, update index and repeat
                await build_index_task

                query_task.cancel()
                client.update_index(index_path)


asyncio.run(main())
