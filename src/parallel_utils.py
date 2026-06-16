from typing import Callable, Any
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm


def jax_joblib_configuration():
    import os
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_cpu_use_xnnpack=false "
        "--xla_cpu_parallel_codegen_split_count=1 "
        "--xla_force_host_platform_device_count=1 "
    )
    #import numpyro
    #numpyro.set_host_device_count(1)


class ProgressParallel(Parallel):

    def __init__(self, use_tqdm=True, total=None, desc="", *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self.desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm,
                  total=self._total,
                  desc=self.desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def apply_in_parallel(function: Callable,
                      data_list: list[Any],
                      n_jobs: int,
                      backend: str = 'loky',
                      description: str = 'Processing',
                      ):
    if not callable(function):
        raise ValueError("function must be callable")
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs has to be greater than 0")
    n_items = len(data_list)
    #if n_jobs == 1: # Defaults to serial execution
    #    return [function(item) for item in tqdm(data_list, desc=description)]
    #if backend == "threading":
    #    jax_joblib_configuration()
    if backend == 'loky':
        with parallel_config(backend='loky', inner_max_num_threads=1):
            with ProgressParallel(
                n_jobs=n_jobs, total=n_items, desc=description, return_as='list', initializer=jax_joblib_configuration
            ) as parallel:
                return parallel(delayed(function)(item) for item in data_list)
    elif backend == 'threading':
        with parallel_config(backend='threading'):
            with ProgressParallel(
                n_jobs=n_jobs, total=n_items, desc=description, return_as='list'
            ) as parallel:
                return parallel(delayed(function)(item) for item in data_list)
    else:
        raise ValueError("Only threading and loky backends are supported")
