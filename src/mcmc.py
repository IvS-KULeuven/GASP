import argparse
import logging
from pathlib import Path
from functools import partial
from typing import Callable
import numpy as np
import jax
from jax import random, vmap, jit, Array
import jax.numpy as jnp
from jax.scipy import linalg
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from numpyro.diagnostics import summary
import numpyro.distributions as dist
import polars as pl

from parallel_utils import apply_in_parallel, jax_joblib_configuration
from preprocessing import preprocess_lc, pack_light_curve
import kernels


logger = logging.getLogger(__name__)

covariance_functions = {
    'red_1': kernels.powered_exponential_covariance,  # gamma=1.0
    'red_15': partial(kernels.powered_exponential_covariance, gamma=1.5),
    'red_2': partial(kernels.powered_exponential_covariance, gamma=2.0),
    'cos': kernels.cosine_covariance,
    'periodic': kernels.periodic_covariance,
    'cos_plus_linear': kernels.cosine_plus_linear_covariance,
    'cos_plus_red': kernels.cosine_plus_exp_covariance,
    'cos_times_red': kernels.cosine_times_exp_covariance,
    'cos_plus_red_ratio': kernels.cosine_plus_exp_ratio_covariance,
}

priors = {
    'red_1': {'red_length', 'red_stdev'},
    'red_15': {'red_length', 'red_stdev'},
    'red_2': {'red_length', 'red_stdev'},
    'cos': ['cos_frequency', 'cos_stdev'],
    'periodic': ['cos_frequency', 'cos_stdev', 'cos_length'],
    'cos_plus_linear': ['cos_frequency', 'cos_stdev', 'red_stdev'],
    'cos_plus_red': ['cos_frequency', 'cos_stdev', 'red_length', 'red_stdev'],
    'cos_plus_red_ratio': ['cos_frequency', 'cos_stdev', 'red_length', 'beta'],
    'cos_times_red': ['cos_frequency', 'cos_stdev', 'red_length']
}


def beta_prior():
    return dist.Uniform(0.001, 0.999)


def length_scale_prior(time):
    ub = time[-1] - time[0]
    lb = jnp.amin(time[1:] - time[:-1])
    return dist.Uniform(lb, ub)


def stdev_prior(scale=1.5):
    return dist.HalfNormal(scale=scale)


def frequency_prior(time, central_frequency):
    T = time[-1] - time[0]
    lb = central_frequency - 0.5/T
    ub = central_frequency + 0.5/T
    return dist.Uniform(lb, ub)


def bayesian_gp_model(x, y, yerr, covariance, period=None, extra_white_noise=True, eps=1e-11):
    N = len(x)
    hyperparameters = {}
    if covariance not in covariance_functions:
        raise ValueError(f"Covariance {covariance} is not implemented.")
    for prior in priors[covariance]:
        if 'stdev' in prior:
            hyperparameters[prior] = numpyro.sample(prior, stdev_prior())
        elif 'frequency' in prior:
            if period is None:
                raise ValueError("Period has to be specified for periodic kernels")
            hyperparameters[prior] = numpyro.sample(
                prior, frequency_prior(x, central_frequency=1.0/period)
            )
        elif prior == 'red_length':
            hyperparameters[prior] = numpyro.sample(prior, length_scale_prior(x))
        elif prior == 'cos_length':
            hyperparameters[prior] = numpyro.sample(prior, dist.Uniform(0.01, 10.0))
        elif prior == 'beta':
            hyperparameters[prior] = numpyro.sample(prior, beta_prior())
        else:
            raise ValueError(f"Prior {prior} is not implemented")
    mean = numpyro.sample("mean", dist.Normal(loc=0.0, scale=1.0))
    C = covariance_functions[covariance](x, x, hyperparameters) + jnp.eye(N) * eps + jnp.diag(jnp.pow(yerr, 2))
    if extra_white_noise:
        white_stdev = numpyro.sample("white_stdev", stdev_prior(0.5))
        C += jnp.eye(N) * jnp.pow(white_stdev, 2)

    numpyro.sample(
        "Y",
        dist.MultivariateNormal(
            loc=mean * jnp.ones(N),
            covariance_matrix=C
        ),
        obs=y,
    )


def run_sampler(rng_key, x, y, yerr, model,
                num_warmup: int = 500,
                num_samples: int = 1000,
                num_chains: int = 4,
                thinning: int = 2,
                verbose: bool = True,
                chain_method: str = 'sequential'):
    sampler = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=verbose,
        chain_method=chain_method,
        jit_model_args=True,
    )

    sampler.run(rng_key, x, y, yerr)
    samples = sampler.get_samples()
    if verbose:
        sampler.print_summary()
    r_hats, n_effs = {}, {}
    log_density_params = partial(
        log_density, model=model, model_args=(x, y, yerr), model_kwargs={}
    )
    log_posterior = jit(vmap(
        lambda params: log_density_params(params=params)[0]
    ))(params=samples)
    for k, v in summary(sampler.get_samples(group_by_chain=True)).items():
        r_hats[f'r_hat_{k}'] = v['r_hat'].item()
        n_effs[f'n_eff_{k}'] = v['n_eff'].item()
    return samples, log_posterior, r_hats, n_effs


@partial(jit, static_argnames=['kernel'])
def predict_gp(rng_key, x, y, x_test, kernel, hyperparameters, eps=1e-11):
    N, N_test = x.shape[0], x_test.shape[0]
    if 'white_stdev' in hyperparameters:
        white_variance = eps + hyperparameters['white_stdev'] ** 2
    else:
        white_variance = eps
    prior_mean = hyperparameters['mean']
    k_pp = kernel(x_test, x_test, hyperparameters) + white_variance * jnp.eye(N_test)
    k_px = kernel(x_test, x, hyperparameters)
    k_xx = kernel(x, x, hyperparameters) + white_variance * jnp.eye(N)

    K_xx_cho = linalg.cho_factor(k_xx)
    post_mean = jnp.matmul(k_px, linalg.cho_solve(K_xx_cho, y-prior_mean)) + prior_mean
    post_cov = k_pp - jnp.matmul(k_px, linalg.cho_solve(K_xx_cho, k_px.T))
    post_scale = jnp.sqrt(jnp.clip(jnp.diag(post_cov), 0.0))
    eps = random.normal(rng_key, x_test.shape[:1])
    return post_mean, post_mean + post_scale * eps


def predict_gp_parallel(
    x: Array,
    y: Array,
    x_test: Array,
    kernel: Callable,
    samples: dict[str, Array],
    rng_key: int,
    ) -> tuple[Array, Array]:
    first_key = next(iter(samples))
    chain_length = len(samples[first_key])
    rng_keys = random.split(rng_key, chain_length)

    return vmap(
        predict_gp, in_axes=(
            0, None, None, None, None, {k: 0 for k in samples.keys()}
        )
    )(
        rng_keys, x, y, x_test, kernel, samples
    )


@partial(jit, static_argnames=['kernel'])
def log_marginal_likelihood(time, mag, kernel, hyperparameters):
    # https://www.cs.helsinki.fi/u/ahonkela/teaching/compstats1/book/multivariate-normal-distributions-and-numerical-linear-algebra.html
    noise_var = hyperparameters['white_stdev'] ** 2
    k = kernel(time, time, hyperparameters) + jnp.eye(len(time))*noise_var
    chol = linalg.cholesky(k, lower=True)
    alpha = linalg.solve_triangular(
        chol, mag - hyperparameters['mean'], lower=True
    )
    halflogdetK = jnp.sum(jnp.log(jnp.diag(chol)))
    const = 0.5 * chol.shape[0] * jnp.log(2 * jnp.pi)
    return -0.5 * jnp.sum(jnp.square(alpha)) - halflogdetK - const


def log_marginal_likelihood_parallel(
    x: Array,
    y: Array,
    kernel: Callable,
    samples: dict[str, Array],
    ) -> Array:
    return vmap(
        log_marginal_likelihood, in_axes=(None, None, None, {k: 0 for k in samples.keys()})
    )(
        x, y, kernel, samples
    )


def batch_sampler(parquet_path: Path, result_dir: Path, model_type: str, rseed: int = 1234):
    jax_joblib_configuration()
    df_lcs = pl.read_parquet(parquet_path)
    rng_key = random.PRNGKey(rseed)
    rng_keys = random.split(rng_key, len(df_lcs))
    save_path = result_dir / f"{parquet_path.stem}.parquet"
    if save_path.exists():
        logger.info(f"File {save_path} exists and will not be overwritten.")
        return None
    df_mcmc = []
    for k in range(len(df_lcs)):
        row = df_lcs.slice(k, 1)
        period = row.select('period').to_series().item()
        df_lc = pack_light_curve(row, remove_extreme_errors=True)['g']
        time, mag, err = preprocess_lc(df_lc, scale_time=False, center_mag=True, scale_mag=True)
        gp_model = partial(bayesian_gp_model, covariance=model_type, period=period)
        samples, log_posterior, r_hats, n_effs = run_sampler(
            rng_keys[k], jnp.asarray(time), jnp.asarray(mag), jnp.asarray(err),
            gp_model, verbose=False
        )
        df_mcmc.append(
            pl.DataFrame(
                {'sourceid': row['sourceid'].item()} | r_hats | n_effs
            ).with_columns(
                pl.lit(np.asarray(log_posterior).reshape(1, -1)).alias('log_posterior'),
                *[pl.lit(np.asarray(v).reshape(1, -1)).alias(k) for k, v in samples.items()],
            )
        )
        for v in samples.values():
            v.delete()
        jax.clear_caches()
    pl.concat(df_mcmc).write_parquet(save_path, compression='zstd', compression_level=22)
    return None


if __name__ == '__main__':
    jax_joblib_configuration()
    parser = argparse.ArgumentParser(description='AGNs posteriors with MCMC')
    parser.add_argument('path_to_dataset', type=str)
    parser.add_argument('save_directory', type=str)
    parser.add_argument('model_type', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    path_dataset = Path(args.path_to_dataset)
    model_type = args.model_type
    save_directory = Path(args.save_directory) / model_type
    save_directory.mkdir(exist_ok=True, parents=True)
    parquet_list = sorted(list((path_dataset).glob('*.parquet')))
    print(f"Found {len(parquet_list)} files, starting run")
    # batch_sampler(parquet_list[0], save_directory)

    apply_in_parallel(
        partial(batch_sampler,
                result_dir=save_directory,
                model_type=model_type),
        parquet_list,
        n_jobs=args.n_jobs,
        description='Running MCMC'
    )
