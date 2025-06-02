import jax
import numpyro
from typing import Callable, Any
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm


def jax_joblib_configuration():
    jax.config.update("jax_platform_name", 'cpu')
    jax.config.update("jax_enable_x64", True)
    numpyro.set_host_device_count(1)


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
                      description: str = 'Processing',
                      ):
    if not callable(function):
        raise ValueError("function must be callable")
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs has to be greater than 0")
    n_items = len(data_list)
    if n_jobs > 1:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            with ProgressParallel(n_jobs=n_jobs,
                                  total=n_items,
                                  desc=description,
                                  return_as='list') as parallel:
                parallel(delayed(function)(item) for item in data_list)
    else:
        # Defaults to serial execution
        [function(item) for item in tqdm(data_list, desc=description)]
