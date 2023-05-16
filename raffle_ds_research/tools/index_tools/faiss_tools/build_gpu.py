from __future__ import annotations

import contextlib
import re
import sys
import time
from typing import Callable, Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import pydantic
import torch
from loguru import logger
from tqdm import tqdm

from raffle_ds_research.tools import dstruct
from raffle_ds_research.tools.index_tools.faiss_tools import support


def _get_gpu_resources(
    devices: list[int], tempmem: int = -1, log_mem_allocation: bool = False
) -> list[faiss.GpuResources]:
    """Return a list of GPU resources."""
    gpu_resources = []
    ngpu = torch.cuda.device_count() if devices is None else len(devices)
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        res.setLogMemoryAllocations(log_mem_allocation)
        if tempmem is not None and tempmem > 0:
            logger.debug(f"Setting GPU:{i} temporary memory to {support.format_bytes(tempmem, 'MB')}")
            res.setTempMemory(tempmem)

        gpu_resources.append(res)

    return gpu_resources


class WithTimer:
    """Context manager to log the time taken for an event."""

    def __init__(self, event_name: str, log: Callable[[str], None]) -> None:
        self.log = log
        self.event_name = event_name
        self._enter_time: Optional[float] = None

    def __enter__(self) -> None:
        self._enter_time = time.time()
        self.log(f"Starting `{self.event_name}`")

    def __exit__(self, *args, **kwargs) -> None:  # noqa: ANN
        if self._enter_time is None:
            raise ValueError("`enter()` was not called")
        self.log(f"Completed `{self.event_name}` in {time.time() - self._enter_time:.2f}s")


class FaissGpuConfig(pydantic.BaseModel):
    """Configuration for training a faiss index on GPUs."""

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = False
        extra = "forbid"

    devices: list[int] = [-1]
    use_float16: bool = True
    use_precomputed_tables: bool = True
    max_add: Optional[int] = 2**18
    tempmem: Optional[int] = -1
    keep_indices_on_cpu: bool = True
    verbose: bool = True
    shard: bool = True
    add_batch_size: int = 2**18

    @pydantic.validator("devices", pre=True, always=True)
    def _validate_devices(cls, v):  # noqa: ANN
        if v is None or v == [-1]:
            return list(range(torch.cuda.device_count()))
        return v

    def cloner_options(self) -> faiss.GpuMultipleClonerOptions:
        """Return a faiss.GpuMultipleClonerOptions."""
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = self.use_float16
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = self.use_precomputed_tables
        if self.keep_indices_on_cpu:
            co.indicesOptions = faiss.INDICES_CPU
        co.verbose = self.verbose
        if self.max_add is not None:
            co.reserveVecs = self.max_add

        co.shard = self.shard

        return co

    def gpu_resources(self) -> list[faiss.GpuResources]:
        """Return a list of GPU resources."""
        if not self.devices:
            raise ValueError(f"devices must be set to use `resource_vectors()`. devices={self.devices}")
        return _get_gpu_resources(self.devices, self.tempmem or -1)


def train_ivfpq_multigpu(
    ivfpq_factory: support.IVFPQFactory,
    *,
    x_train: np.ndarray,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
    gpu_config: Optional[FaissGpuConfig] = None,
) -> tuple[None | faiss.VectorTransform, faiss.Index]:
    """Train a faiss index using multiple GPUs."""
    gpu_config = gpu_config or FaissGpuConfig()

    # make the preprocessor
    with WithTimer(f"Training processor {ivfpq_factory.preproc}", logger.info):
        preprocessor = (
            _train_preprocessor(x_train=x_train, preproc=ivfpq_factory.preproc) if ivfpq_factory.preproc else None
        )

    # train the IVF index (coarse quantizer / centroids)
    with WithTimer(f"Training IVF coarse quantizer with {ivfpq_factory.n_centroids} centroids", logger.info):
        ivf = _train_ivf(
            x_train,
            n_centroids=ivfpq_factory.n_centroids,
            preprocessor=preprocessor,
            config=gpu_config,
            faiss_metric=faiss_metric,
        )

    with WithTimer(
        f"Training PQ refiner `{ivfpq_factory.ncodes}x{ivfpq_factory.nbits}` ({ivfpq_factory.encoding})", logger.info
    ):
        ivfpq = _train_ivfpq(
            x_train,
            ivfpq_factory=ivfpq_factory,
            preprocessor=preprocessor,
            ivf=ivf,
            faiss_metric=faiss_metric,
        )
    return preprocessor, ivfpq


def _train_preprocessor(x_train: np.ndarray, *, preproc: str) -> faiss.VectorTransform:
    """Train a faiss preprocessor VectorTransform."""
    if preproc.startswith("OPQ"):
        opq_pattern = re.compile(r"OPQ(\d+)(?:_(\d+))?")  # noqa: F821
        opq_match = opq_pattern.match(preproc)
        if opq_match is None:
            raise ValueError(f"Invalid OPQ preprocessor: `{preproc}`")
        m = int(opq_match.group(1))  # noqa: F821
        dout = int(opq_match.group(2)) if opq_match.group(2) else x_train.shape[1]  # noqa: F821
        preproc_op = faiss.OPQMatrix(x_train.shape[1], m, dout)
    elif preproc.startswith("PCAR"):
        dout = int(preproc[4:-1])
        preproc_op = faiss.PCAMatrix(x_train.shape[1], dout, 0, True)
    else:
        raise ValueError(f"Unknown preprocessor: `{preproc}`")

    # train and return
    preproc_op.train(x_train)  # type: ignore
    return preproc_op  # type: ignore


def _train_ivf(
    x_train: np.ndarray,
    *,
    n_centroids: int,
    config: FaissGpuConfig,
    preprocessor: None | faiss.VectorTransform,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
    max_points_per_centroid: Optional[int] = 10_000_000,
    verbose: bool = True,
) -> faiss.IndexFlat:
    """Train the centroids for the IVF index."""
    logger.debug(f"About to train IVF (x_train={x_train.shape}")
    d = preprocessor.d_out if preprocessor else x_train.shape[1]
    cluster = faiss.Clustering(d, n_centroids)
    cluster.verbose = verbose
    if max_points_per_centroid is not None:
        cluster.max_points_per_centroid = max_points_per_centroid

    # preprocess te vectors
    if preprocessor is not None:
        x_train = preprocessor.apply(x_train)  # type: ignore

    # make a flat index
    flat_index = faiss.IndexFlat(d, faiss_metric)

    # move the index to CUDA
    with WithTimer("IVF: Moving clusters to GPU", logger.debug):
        gpu_resources = config.gpu_resources()
        index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, flat_index)  # type: ignore

    # train the clustering
    with WithTimer("IVF: clustering", logger.debug):
        cluster.train(x_train, index)  # type: ignore
        centroids = faiss.vector_float_to_array(cluster.centroids)
        centroids = centroids.reshape(n_centroids, d)

    # return the corresponding index
    with WithTimer("IVF: Converting clusters to flat index.", logger.debug):
        coarse_quantizer = faiss.IndexFlat(d, faiss_metric)
        coarse_quantizer.add(centroids)  # type: ignore

    return coarse_quantizer


def _train_index_on_gpu(x_train: np.ndarray, index: faiss.Index) -> faiss.Index:
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    gpu_index.train(x_train)
    return faiss.index_gpu_to_cpu(gpu_index)


def _train_ivfpq(
    x_train: np.ndarray,
    *,
    ivfpq_factory: support.IVFPQFactory,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
    preprocessor: None | faiss.VectorTransform,
    ivf: faiss.IndexFlat,
) -> faiss.Index:
    """Build the full IVF index."""
    cls = {
        "flat": lambda: faiss.IndexIVFFlat,
        "": lambda: faiss.IndexIVFPQ,
        "fs": lambda: faiss.IndexIVFPQFastScan,  # type: ignore
        "fsr": lambda: faiss.IndexIVFPQFastScanResidual,  # type: ignore
    }[ivfpq_factory.encoding]()

    # build the index
    logger.debug("About to train composite index")
    if cls == faiss.IndexIVFFlat:
        index = cls(ivf, ivf.d, ivf.ntotal, faiss_metric)
    else:
        index = cls(
            ivf,
            ivf.d,
            ivf.ntotal,
            ivfpq_factory.ncodes,
            ivfpq_factory.nbits,
            faiss_metric,
        )

    # move the ownership of the index to the new IVF index
    ivf.this.disown()  # type: ignore
    index.own_fields = True

    if isinstance(index, faiss.IndexIVFFlat):
        return index

    # finish training on CPU
    if preprocessor is not None:
        x_train = preprocessor.apply(x_train)  # type: ignore

    # with WithTimer("Training fine quantizer on CPU..", logger.debug):
    # index.train(x_train)  # type: ignore

    with WithTimer("Training fine quantizer on GPU..", logger.debug):
        index = _train_index_on_gpu(x_train, index)

    return index


def build_faiss_index_multigpu(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    factory_string: str,
    train_size: Optional[int] = None,
    faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
    gpu_config: Optional[FaissGpuConfig] = None,
) -> faiss.Index:
    """Build a faiss IVF-PQ index using multiple GPUs."""
    gpu_config = gpu_config or FaissGpuConfig()

    # parse the factory string
    logger.info(f"Building index `{factory_string}`.")
    ivfpq_factory = support.IVFPQFactory.parse(factory_string)

    # build and train the index
    x_train = _sample_train_vecs(vectors, train_size)
    logger.info(f"Training index with {len(x_train)} vectors.")
    preprocessor, index = train_ivfpq_multigpu(
        ivfpq_factory=ivfpq_factory,
        x_train=x_train,
        faiss_metric=faiss_metric,
        gpu_config=gpu_config,
    )

    if not index.is_trained:
        raise RuntimeError("The faiss index is not trained.")

    # populate the index
    logger.info(f"Populating index with {len(vectors)} vectors.")
    try:
        index = _populate_index_multigpu(
            vectors,
            index=index,
            preprocessor=preprocessor,
            gpu_config=gpu_config,
        )
    except RuntimeError as e:
        logger.warning(f"Failed to populate index: {e}")
        logger.info("Trying to populate index on CPU.")
        _populate_index_cpu(vectors, preprocessor=preprocessor, index=index)

    if index.ntotal != len(vectors):
        raise RuntimeError(f"Index has {index.ntotal} vectors, but the dataset has {len(vectors)}.")

    # warp the index with the preprocessor
    if preprocessor is not None:
        logger.info("Wrapping index with preprocessor.")
        index = faiss.IndexPreTransform(preprocessor, index)

    return index


def _sample_train_vecs(vectors: dstruct.SizedDataset[np.ndarray], train_size: Optional[int]) -> np.ndarray:
    if train_size is None or train_size >= len(vectors):
        return vectors[:]

    ids = np.random.choice(len(vectors), size=train_size, replace=False)
    return vectors[ids]


def _populate_index_cpu(
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    index: faiss.Index,
    preprocessor: Optional[faiss.VectorTransform] = None,
    max_add: int = 10_000,
) -> faiss.Index:
    """Add elements to a sharded index."""
    nb = len(vectors)
    steps = range(0, nb, max_add)
    slices = (slice(i0, min(nb, i0 + max_add)) for i0 in steps)
    vectors_batch_iter = support.rate_limited_imap(
        lambda idx: (
            idx,
            preprocessor.apply(vectors[idx]) if preprocessor is not None else vectors[idx],  # type: ignore
        ),
        seq=slices,
    )

    # add the vectors
    for _, xs in tqdm(vectors_batch_iter, desc="Adding vectors to sharded index", total=len(steps)):
        index.add(xs)

    return index


def _populate_index_multigpu(  # noqa: PLR0912
    vectors: dstruct.SizedDataset[np.ndarray],
    *,
    index: faiss.Index,
    gpu_config: FaissGpuConfig,
    preprocessor: Optional[faiss.VectorTransform] = None,
) -> faiss.Index:
    """Add elements to a sharded index."""
    co = gpu_config.cloner_options()
    gpu_resources = gpu_config.gpu_resources()
    ngpu = len(gpu_resources)
    max_add = gpu_config.max_add * max(1, ngpu) if gpu_config.max_add is not None else len(vectors)

    # move the cpu index to GPU
    with WithTimer(f"Moving full index to GPU shars ({ngpu} GPUs).", logger.debug):
        gpu_index: faiss.IndexShards = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index, co)  # type: ignore

    # create an iterator over the vectors
    nb = len(vectors)
    steps = range(0, nb, gpu_config.add_batch_size)
    slices = (slice(i0, min(nb, i0 + gpu_config.add_batch_size)) for i0 in steps)
    vectors_batch_iter = support.rate_limited_imap(
        lambda idx: (
            idx,
            preprocessor.apply(vectors[idx]) if preprocessor is not None else vectors[idx],  # type: ignore
        ),
        seq=slices,
    )

    # add the vectors
    with WithTimer(f"Populating index ({steps} steps)", logger.info):
        for i_slice, xs in tqdm(vectors_batch_iter, desc="Adding vectors to sharded index", total=len(steps)):
            i0 = i_slice.start
            i1 = i0 + xs.shape[0]
            if np.isnan(xs).any():
                logger.warning(f"NaN detected in vectors {i0}-{i1}")
                xs[np.isnan(xs)] = 0

            gpu_index.add_with_ids(xs, np.arange(i0, i1))
            if gpu_index.ntotal > max_add:
                # logger.debug(f"Reached max. size per GPU ({max_add}). Flushing indices to CPU")
                if ngpu > 1:
                    for i in range(ngpu):
                        index_src_gpu = faiss.downcast_index(gpu_index.at(i))
                        index_src = faiss.index_gpu_to_cpu(index_src_gpu)
                        index_src.copy_subset_to(index, 0, 0, nb)
                        index_src_gpu.reset()
                        index_src_gpu.reserveMemory(max_add)
                else:
                    index_src = faiss.index_gpu_to_cpu(gpu_index)
                    index_src.copy_subset_to(index, 0, 0, nb)
                    gpu_index.reset()
                    gpu_index.reserveMemory(max_add)
                try:
                    gpu_index.sync_with_shard_indexes()
                except AttributeError:
                    with contextlib.suppress(AttributeError):
                        gpu_index.syncWithSubIndexes()

            sys.stdout.flush()

    if hasattr(gpu_index, "at"):
        # it is a sharded index
        for i in range(ngpu):
            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
            logger.debug("  - index %d size %d" % (i, index_src.ntotal))
            index_src.copy_subset_to(index, 0, 0, nb)
    else:
        # simple index
        index_src = faiss.index_gpu_to_cpu(gpu_index)
        index_src.copy_subset_to(index, 0, 0, nb)

    del gpu_index
    return index
