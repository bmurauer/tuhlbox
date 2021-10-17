from time import perf_counter
from typing import Callable

from sklearn.datasets import fetch_20newsgroups

from tuhlbox.stringkernels import (
    intersection_kernel,
    legacy_intersection_kernel,
    legacy_presence_kernel,
    legacy_spectrum_kernel,
    presence_kernel,
    spectrum_kernel,
)

data = fetch_20newsgroups()["data"][:100]


def benchmark(kernel_method: Callable) -> float:
    start = perf_counter()
    kernel_method(1, 4)(data, data)
    return perf_counter() - start


kernels = [
    intersection_kernel,
    legacy_intersection_kernel,
    presence_kernel,
    legacy_presence_kernel,
    spectrum_kernel,
    legacy_spectrum_kernel,
]

for kernel in kernels:
    print(f"{kernel.__name__}: {benchmark(kernel)}")
