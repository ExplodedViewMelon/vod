from time import perf_counter, sleep
import numpy as np


class Timer:
    def __init__(self) -> None:
        self.t0: float = 0
        self.t1: float = 0
        self.timings = []

    def begin(self) -> None:
        self.t0 = perf_counter()

    def end(self) -> None:
        self.t1 = perf_counter()
        self.timings.append(self.t1 - self.t0)

    @property
    def mean(self) -> float:
        return float(np.mean(self.timings))

    def pk_latency(self, k) -> float:
        return np.percentile(self.timings, k)

    def __str__(self) -> str:
        return f"{self.mean}s"
