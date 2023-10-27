import os
import requests
import abc
import h5py
import numpy as np
from numpy import ndarray
from tqdm import tqdm


class DatasetHDF5Simple(abc.ABC):
    # gotten from https://github.com/erikbern/ann-benchmarks/tree/main#data-sets
    def __init__(self) -> None:
        self._base_path = os.getcwd()
        self._folder_path: str = f"{self._base_path}/data_benchmark/{self.name}"
        self._file_path: str = f"{self._folder_path}/{self.name}.hdf5"
        self._file_url = f"http://ann-benchmarks.com/{self.name}.hdf5"

        if not self._cache_exists():
            self._download_to_cache()
        self.vectors: ndarray = self._load_from_cache()

    def _cache_exists(self) -> bool:
        return os.path.isfile(self._file_path)

    def _download_to_cache(self) -> None:
        print(f"Downloading {self.name} from {self._file_url}")
        if not os.path.exists(self._folder_path):
            os.makedirs(self._folder_path)

        # download content with context bar
        response = requests.get(self._file_url, stream=True)
        progress_bar = tqdm(total=int(response.headers.get("content-length", 0)), unit="B", unit_scale=True)

        with open(self._file_path, "wb") as file:
            for data in response.iter_content(chunk_size=4096):
                file.write(data)
                progress_bar.update(len(data))  # Update the progress bar

        progress_bar.close()

    def _load_from_cache(self) -> ndarray:
        print(f"Loading {self.name} from cache")
        with h5py.File(self._file_path, "r") as hdf5_file:
            return self._unpack_hdf5_file(hdf5_file)

    def _unpack_hdf5_file(self, hdf5_file: h5py.File) -> ndarray:
        # NOTE: might have to be overloaded for some datasets.
        _train = np.array(hdf5_file["train"])
        _test = np.array(hdf5_file["test"])
        return np.concatenate((_train, _test), axis=0)

    def get_indices_and_queries_split(
        self, n_query_vectors: int, query_batch_size: int, size_limit: int = 0
    ) -> tuple[ndarray, ndarray]:
        n, d = self.vectors.shape
        _test_indices = np.random.choice(n, n_query_vectors * query_batch_size, replace=False)
        query_vectors = self.vectors[_test_indices].reshape((n_query_vectors, query_batch_size, d))
        index_vectors = np.delete(self.vectors, _test_indices, axis=0)
        if size_limit:
            index_vectors = index_vectors[:size_limit]
        return index_vectors, query_vectors

    @property
    @abc.abstractmethod
    def name(self) -> str:
        NotImplementedError()


class DatasetGlove(DatasetHDF5Simple):
    """Dataset of size (~1_000_000, 25) at ~121MB"""

    @property
    def name(self):
        return "glove-25-angular"


class DatasetLastFM(DatasetHDF5Simple):
    """Dataset of size (~350_000, 65) at ~135MB"""

    @property
    def name(self):
        return "lastfm-64-dot"


class DatasetSift1M(DatasetHDF5Simple):
    """Dataset of size (~1_000_000, 128) at ~501MB"""

    @property
    def name(self):
        return "sift-128-euclidean"


# if __name__ == "__main__":
#     # test
#     dataset = DatasetSift1M()
#     print(dataset.vectors.shape)
