import os
import sys
import time
from copy import copy
from pathlib import Path

import numpy as np
import requests
import rich
import vod_types as vt
from vod_search import base, faiss_search, io
from vod_search.socket import find_available_port

from vod_search.models import *

import faiss
import tempfile
import os

# get the path to the server script
server_run_path = Path(__file__).parent / "server.py"


class FaissClient(base.SearchClient):
    """Faiss client for interacting for spawning a Faiss server and querying it."""

    def __init__(
        self,
        host: str = "http://localhost",
        # port: int = 7678,
        port: int = 6637,
    ):
        self.host = host
        self.port = port

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.url}](requires_vectors={self.requires_vectors})"

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    def ping(self, timeout: float = 120) -> bool:
        """Ping the server."""
        try:
            response = requests.get(f"{self.url}/", timeout=timeout)
        except requests.exceptions.ConnectionError:
            return False

        response.raise_for_status()
        return "OK" in response.text

    def update_index(self, index_path: str, timeout=600) -> bool:
        """Update the index path."""
        response = requests.post(
            f"{self.url}/update",
            json={"index_path": index_path},
            timeout=timeout,
        )
        response.raise_for_status()
        return "OK" in response.text

    # def search_py(self, query_vec: np.ndarray, top_k: int = 3, timeout: float = 120) -> vt.RetrievalBatch:
    #     """Search the server given a batch of vectors (slow implementation)."""
    #     response = requests.post(
    #         f"{self.url}/search",
    #         json={
    #             "vectors": query_vec.tolist(),
    #             "top_k": top_k,
    #         },
    #         timeout=timeout,
    #     )
    #     response.raise_for_status()
    #     data = response.json()
    #     return vt.RetrievalBatch.cast(
    #         indices=data["indices"],
    #         scores=data["scores"],
    #     )

    def search(
        self,
        *,
        vector: np.ndarray,
        text: None | list[str] = None,  # noqa: ARG002
        subset_ids: None | list[list[base.SubsetId]] = None,  # noqa: ARG002
        ids: None | list[list[base.SectionId]] = None,  # noqa: ARG002
        shard: None | list[base.ShardName] = None,  # noqa: ARG002
        top_k: int = 3,
        timeout: float = 120,
    ) -> vt.RetrievalBatch:
        """Search the server given a batch of vectors."""
        start_time = time.time()
        serialized_vectors = io.serialize_np_array(vector)
        payload = {
            "vectors": serialized_vectors,
            "top_k": top_k,
        }
        response = requests.post(f"{self.url}/fast-search", json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            try:
                rich.print(response.json()["detail"])
            except Exception:
                rich.print(response.text)
            raise exc

        data = response.json()
        indices_list = io.deserialize_np_array(data["indices"])
        scores_list = io.deserialize_np_array(data["scores"])

        try:
            return vt.RetrievalBatch.cast(
                indices=indices_list,
                scores=scores_list,
                labels=None,
                meta={"time": time.time() - start_time},
            )
        except Exception as exc:
            rich.print({"indices": indices_list, "scores": scores_list})
            raise exc


class FaissMaster(base.SearchMaster[FaissClient]):
    """The Faiss master client is responsible for spawning and killing the Faiss server.

    ```python
    with FaissMaster(index_path, nprobe=8, logging_level="critical") as client:
        # do stuff with the client
        result = client.search(...)
    ```
    """

    index_parameters: IndexParameters
    _allow_existing_server: bool = True

    def __init__(  # noqa: PLR0913
        self,
        vectors,
        index_parameters,
        logging_level: str = "DEBUG",
        host: str = "http://localhost",
        port: int = 6637,
        skip_setup: bool = False,
        free_resources: bool = False,
        serve_on_gpu: bool = False,
        run_as_docker_image: bool = True,
    ):
        super().__init__(skip_setup=skip_setup, free_resources=free_resources)
        self.vectors = vectors
        self.index_parameters = index_parameters
        self.logging_level = logging_level
        self.host = host
        if port < 0:
            port = find_available_port()
        self.port = port
        self.serve_on_gpu = serve_on_gpu
        self.run_as_docker_image = run_as_docker_image
        self._build_index()

    def _make_env(self) -> dict[str, str]:
        env = copy(dict(os.environ))  # type: ignore
        if "KMP_DUPLICATE_LIB_OK" not in env:
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["LOGURU_LEVEL"] = self.logging_level.upper()
        # add the local path, so importing the library will work.
        if "PYTHONPATH" not in env:
            env["PYTHONPATH"] = str(Path.cwd())
        else:
            os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{Path.cwd()}"
        return env

    def _make_cmd(self) -> list[str]:
        executable_path = sys.executable  # TODO also pass ef_search
        if self.run_as_docker_image:
            return [
                "docker",
                "run",
                "--name",
                "faiss-server",
                "--rm",
                "-p",
                f"{self.port}:{self.port}",
                "-v",
                f"{self.tmpdir.name}:/vod/faiss_index",
                "-t",
                "faiss_server",
            ]
        else:
            return [
                str(executable_path),
                str(server_run_path),
                "--index-path",
                str(self.index_path),
                "--nprobe",
                (
                    str(self.index_parameters.index_type.n_probe)
                    if isinstance(self.index_parameters.index_type, IVF)
                    else "0"
                ),
                "--host",
                str(self.host),
                "--port",
                str(self.port),
                "--logging-level",
                str(self.logging_level),
                *(["--serve-on-gpu"] if self.serve_on_gpu else []),
            ]

    def get_client(self) -> FaissClient:
        """Get the client for interacting with the Faiss server."""
        return FaissClient(host=self.host, port=self.port)

    @property
    def url(self) -> str:
        """Return the URL of the server."""
        return f"{self.host}:{self.port}"

    @property
    def service_info(self) -> str:
        """Return the name of the service."""
        return f"FaissServer[{self.url}]"

    @property
    def service_name(self) -> str:
        """Return the name of the service."""
        return super().service_name + f"-{self.port}"

    def _get_factory_string(self):
        preprocessing: None | ProductQuantization | ScalarQuantization = self.index_parameters.preprocessing
        index_type: HNSW | IVF = self.index_parameters.index_type

        if isinstance(index_type, IVF):  # TODO move n_probe to search
            if isinstance(preprocessing, ProductQuantization):
                # return f"IVF{index_type.n_partition},PQ{preprocessing.m}x{preprocessing.n_bits}" # this does work but nbits are temp. dropped.
                return f"IVF{index_type.n_partition},PQ{preprocessing.m}"
            elif isinstance(preprocessing, ScalarQuantization):
                return f"IVF{index_type.n_partition},SQ{preprocessing.n}"
            else:  # preprocessing is None
                return f"IVF{index_type.n_partition},Flat"

        if isinstance(index_type, HNSW):
            if isinstance(preprocessing, ProductQuantization):
                return f"HNSW{index_type.M},PQ{preprocessing.m}"
            elif isinstance(preprocessing, ScalarQuantization):
                return f"HNSW{index_type.M},SQ{preprocessing.n}"
            else:  # preprocessing is None
                return f"HNSW{index_type.M},Flat"

            # TODO include ef_construction and ef_search.

    def _build_index(self) -> None:
        self.timerBuildIndex.begin()
        factory_string = self._get_factory_string()
        # index = faiss.index_factory(self.vectors.shape[-1], factory_string)

        # index.add(self.vectors)

        index = faiss_search.build_faiss_index(  # valentin's code
            vectors=self.vectors,
            factory_string=factory_string,
            ef_construction=(
                self.index_parameters.index_type.ef_construction
                if isinstance(self.index_parameters.index_type, HNSW)
                else -1
            ),
        )

        if isinstance(self.index_parameters.index_type, HNSW):
            index.hnsw.efSearch = self.index_parameters.index_type.ef_search  # type: ignore

        self.tmpdir = tempfile.TemporaryDirectory()
        self.index_path = f"{self.tmpdir.name}/index.faiss"
        faiss.write_index(index, self.index_path)
        self.timerBuildIndex.end()

    def _cleanup(self):
        print("Exiting and cleaning up temporary data folder")
        self.tmpdir.cleanup()  # clean up the temporary folder

    def _on_exit(self) -> None:
        self._cleanup()
        return super()._on_exit()

    def __repr__(self) -> str:
        return f"index: faiss {self.index_parameters}"
