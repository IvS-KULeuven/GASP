from pathlib import Path
import argparse
from functools import partial
import numpy as np
import jax
from jax import jit, random, vmap
import jaxopt
from tinygp import GaussianProcess, kernels
import polars as pl
from nifty_ls import finufft
from parallel_utils import apply_in_parallel

from jax import numpy as jnp
from jax.nn import sigmoid
from tinygp.helpers import JAXArray
from tinygp.kernels import Stationary
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary

def unpack_light_curve(df_row: pl.DataFrame) -> dict[str, np.ndarray]:
    def col_to_array(col: pl.Series) -> np.ndarray:
        return col[0].to_numpy().astype('float64')
    light_curve = {}
    bands = [col.split('_')[0] for col in df_row.columns if 'obstimes' in col]
    for band in bands:
        time = col_to_array(df_row[f'{band}_obstimes'])
        val = col_to_array(df_row[f'{band}_val'])
        valerr = col_to_array(df_row[f'{band}_valerr']) 
        light_curve[band] = np.stack([time, val, valerr])
    return light_curve

def clean_light_curve(lc, remove_extreme_errors: bool = True, remove_extreme_values: bool = True) -> dict[str, np.ndarray]:
    clean_light_curve = {}
    for band, lcb in lc.items():
        valerr = lcb[-1]
        mask = ~np.isinf(valerr) & ~np.isnan(valerr)
        lcb = lcb[:, mask]
        if remove_extreme_errors:
            valerr = lcb[-1]
            outlier_abs = valerr > 0.5
            iqr_err = np.subtract(*np.percentile(valerr, (75, 25)))
            outlier_rel = valerr  > np.median(valerr) + 12*iqr_err
            outlier = outlier_abs | outlier_rel
            lcb = lcb[:, ~outlier]
        if remove_extreme_values:
            val = lcb[-2]
            iqr_val = np.subtract(*np.percentile(val, (75, 25)))
            outlier = np.abs(val - np.median(val)) > 2*iqr_val
            lcb = lcb[:, ~outlier]            
        idx = lcb[0].argsort()
        clean_light_curve[band] = lcb[:, idx]
    return clean_light_curve

class PoweredExp(Stationary):

    gamma: JAXArray | float | None = None

    def __check_init__(self):
        if self.gamma is None:
            raise ValueError("Missing required argument 'gamma'")

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        return jnp.exp(-r ** self.gamma)

def box_constraint(eta, min_val, max_val):
    return min_val + (max_val-min_val)*sigmoid(eta)

def build_gp(theta, X, Yerr, kernel_fn, mean_fn=None):
    return _build_gp(
        X=X, Yerr=Yerr, log_sigma=theta['log_sigma'], log_tau=theta['log_tau'], mean=theta['mean'],
        kernel_fn=kernel_fn, mean_fn=mean_fn, logit_gamma=theta['logit_gamma'] if 'logit_gamma' in theta else None,
    )


def _build_gp(X, Yerr, log_sigma, log_tau, mean, kernel_fn, logit_gamma=None, mean_fn=None):
    sigma = jnp.exp(log_sigma)
    tau = jnp.exp(log_tau)
    if logit_gamma is not None:
        gamma = box_constraint(logit_gamma, 1.0, 2.0)
        kernel = sigma**2 * kernel_fn(tau, gamma=gamma)
    else:
        kernel = sigma**2 * kernel_fn(tau)
    if mean_fn is not None:
        mean = partial(mean_fn, mean)
    return GaussianProcess(kernel, X, diag=Yerr**2, mean=mean)


def find_local_maxima(x: np.ndarray,
                      how_many: int,
                      ) -> np.ndarray:
    local_maxima = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.argsort(x[1:-1][local_maxima])[::-1][:how_many]
    return np.arange(1, len(x)-1)[local_maxima][idx]


def compute_periodogram(time, mag, err, fmin: float, fmax: float, fres: float, normalization: str='standard'):
    periodogram = partial(
        finufft.lombscargle,
        t=time, y=mag, dy=err, nthreads=1, normalization=normalization,
    )
    Nf = int((fmax - fmin) / fres)
    freqs = fmin + np.arange(Nf) * fres
    ampls = periodogram(fmin=fmin, df=fres, Nf=len(freqs))
    return freqs, ampls

def fit_gp(time, mag, err, period: float, n_harmonics: int, init_scale: float = 100.0, init_logit_gamma: float = 0., fit_gamma: bool = True):
    if n_harmonics > 0:
        initial_mean = jnp.zeros(shape=(n_harmonics*2 + 1,))
        initial_mean[0] = jnp.mean(mag)
        mean_fn = partial(fourier_series_mean_function, period=period, n_harmonics=n_harmonics)
    else:
        initial_mean = jnp.mean(mag)
        mean_fn = None
    theta_init = {
        "mean": initial_mean,
        "log_sigma": jnp.log(jnp.std(mag)),
        "log_tau": jnp.log(init_scale),
    }
    if fit_gamma:
        theta_init['logit_gamma'] = init_logit_gamma
    build_gp_ = partial(build_gp, X=time, Yerr=err, kernel_fn=PoweredExp if fit_gamma else kernels.Exp, mean_fn=mean_fn)
   
    @jax.jit
    def loss(params):
        return -build_gp_(params).log_probability(mag)
        
    solver = jaxopt.ScipyMinimize(fun=loss, jit=True)
    soln = solver.run(theta_init)
    gp = build_gp_(soln.params)
    return soln, gp

def fit_gp_red_noise(time, mag, err, init_scale: float = 100., init_logit_gamma: float = 0., fit_gamma=True):
    soln, gp = fit_gp(
        jnp.asarray(time), jnp.asarray(mag), jnp.asarray(err), 
        n_harmonics=0, period=1.0, init_scale=init_scale, init_logit_gamma=init_logit_gamma, fit_gamma=fit_gamma,
    )
    return gp, soln.params, soln.state.fun_val


def _weighted_chi2_constant(y, sigma):
    w = 1.0 / (sigma ** 2)
    ybar = jnp.sum(w * y) / jnp.sum(w)
    r = (y - ybar) / sigma
    return jnp.sum(r * r)

"""
def sine_wave_model(time, period, params=None):
    A = jnp.column_stack(
        [jnp.ones_like(time),
         jnp.cos(2.*jnp.pi*time/period),
         jnp.sin(2.*jnp.pi*time/period)]
    )
    if params is not None:
        return A, jnp.dot(A, params)
    return A
"""

def fourier_design_matrix(time, period, n_harmonics):
    return jnp.array(
        [jnp.cos(2.*jnp.pi*k*time/period) for k in range(0, n_harmonics+1)] +
        [jnp.sin(2.*jnp.pi*k*time/period) for k in range(1, n_harmonics+1)]
    )

def fourier_series_mean_function(params, time, period, n_harmonics):
    A = fourier_design_matrix(time, period=period, n_harmonics=n_harmonics)
    return params @ A.T

@partial(jit, static_argnames=['return_params', 'standardize_data'])
def fit_sine_wave(x, y, yerr, period, return_params=False, standardize_data=False):
    if standardize_data:
        loc, scale = jnp.mean(y), jnp.std(y)
        y = (y - loc)/scale
        yerr = yerr/scale
    A = fourier_design_matrix(x, period, n_harmonics=1).T
    Aw = A / yerr[:, None]
    Bw = y / yerr
    params, _, _, _ = jnp.linalg.lstsq(Aw, Bw, rcond=None)
    residual = Aw @ params - Bw
    chi2 = jnp.sum(residual * residual)
    chi0 = _weighted_chi2_constant(y, yerr)
    stat = 1.0 - chi2/chi0 

    if return_params:
        AtWA = Aw.T @ Aw
        AtWA_inv = jnp.linalg.inv(AtWA)
        n, p = Aw.shape
        s2 = chi2 / (n - p)
        cov_params = s2 * AtWA_inv
        return stat, params, cov_params
    return stat

def bayesian_red_noise(x, yerr, y, x_interp=None):
    smallest_dt = jnp.amin(x[1:] - x[:-1])
    timespan = x[-1] - x[0]
    mean = numpyro.sample("mean", dist.Normal(jnp.mean(y), 0.5))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(jnp.log(jnp.std(y)), 0.5))
    log_tau = numpyro.sample("log_tau", dist.TruncatedNormal(jnp.log(100.), 1.0, low=jnp.log(smallest_dt), high=jnp.log(timespan)))
    logit_gamma = numpyro.sample("logit_gamma", dist.Normal(0, 1))
    gp = jax.jit(_build_gp, static_argnames=['kernel_fn', 'mean_fn'])(X=x, Yerr=yerr**2, log_sigma=log_sigma, log_tau=log_tau, mean=mean, kernel_fn=PoweredExp, logit_gamma=logit_gamma)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)
    if y is not None and x_interp is not None:
        numpyro.deterministic("pred", gp.condition(y, x_interp).gp.loc)

def mcmc_red_noise(time, mag, err, 
                   num_warmup: int = 500, num_samples: int = 2000, num_chains: int = 4):
    nuts_kernel = NUTS(bayesian_red_noise, target_accept_prob=0.9, dense_mass=True, max_tree_depth=(5, 10))
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False, jit_model_args=True, chain_method='sequential')
    rng_key = random.PRNGKey(55873)
    mcmc.run(rng_key, x=jnp.asarray(time), y=jnp.asarray(mag), yerr=jnp.asarray(err))
    samples = mcmc.get_samples()
    r_hats, n_effs = {}, {}
    for k, v in summary(mcmc.get_samples(group_by_chain=True)).items():
        r_hats[f'r_hat_{k}'] = v['r_hat'].item()
        n_effs[f'n_eff_{k}'] = v['n_eff'].item()
    return samples, r_hats, n_effs

def sugeves_fap(alphas, time, mag, err, fmin, fmax, fres, seed=1234, num_reps=1000):
    gp, params, mle = fit_gp_red_noise(time, mag, err)
    params = {k: v.tolist() for k, v in params.items()}
    params['mle'] = mle
    gp_prior_samples = gp.sample(random.PRNGKey(seed), shape=(num_reps,))
    dist_maxima = np.array([np.amax(
        compute_periodogram(time, np.asarray(gp_prior_sample), err, fmin, fmax, fres)[1]
    ) for gp_prior_sample in gp_prior_samples])
    return {str(alpha): np.quantile(dist_maxima, q=1-alpha, axis=0).tolist() for alpha in alphas} | params

def fit_fourier_series_plus_red_noise(time, mag, err, freq, init_scale: float = 100., init_logit_gamma: float = 0., n_harmonics:int = 1):
    soln, _ = fit_gp(
        jnp.asarray(time), jnp.asarray(mag), jnp.asarray(err), 
        period=1./freq, n_harmonics=n_harmonics, init_scale=init_scale, init_logit_gamma=init_logit_gamma
    )
    params = {k: v.tolist() for k, v in soln.params.items()}
    params['mle'] = soln.state.fun_val
    return params

from enum import StrEnum

class Task(StrEnum):
    PERIODOGRAM_MAXIMA = "periodogram_maxima"
    FIT_FOURIER_SERIES_MLE = "fourier_mle"
    FIT_RED_NOISE_MLE = "red_noise_mle"
    FIT_RED_NOISE_MCMC = "red_noise_mcmc"
    MONTE_CARLO_FAP = "monte_carlo_fap"

def extract_from_parquet(parquet_path: Path,
                         save_dir: Path,
                         task: Task,
                         overwrite: bool = False,
                         fres: float = 1e-4,
                         ) -> None:
    jax.config.update("jax_enable_x64", True)
    (save_dir / task.value).mkdir(exist_ok=True, parents=True)
    write_path = save_dir / task.value / parquet_path.name
    if not overwrite and (write_path).exists():
        return None
    df = pl.read_parquet(parquet_path)
    if df.height == 0:
        return None
    """

    if task is Task.BALUEV_FAP:
        # I need the best frequencies from PERIODOGRAM_MAXIMA
        df = df.join(
            pl.read_parquet(
                save_dir / 'periodogram_maxima' / parquet_path.name,
                columns=['sourceid', 'best_frequencies']
            ),
            on='sourceid'
        )
    """
    if task is Task.FIT_FOURIER_SERIES_MLE:
        for col in ['best_frequencies', 'best_amplitudes']:
            if col not in df.columns:
                raise ValueError(f"Task {task} needs column {col}.")
    result = []
    for k in range(df.height):
        row = df.slice(k, 1)
        lc = clean_light_curve(unpack_light_curve(row))
        # lc = pack_light_curve(row, remove_extreme_errors=True)
        time, mag, err = lc['g']
        timespan = time[-1] - time[0]
        sid = row['sourceid'][0]
        if task is Task.PERIODOGRAM_MAXIMA:
            simple_features = {
                'sourceid': sid,
                'num_obs': len(mag),
                'magnitude_mean': np.mean(mag),
                'magnitude_std': np.std(mag),
                'time_duration': timespan
                }
            #freqs, ampls = compute_periodogram(time, mag, err, fmin=7e-4, fmax=25.0, fres=1e-5)
            freqs, ampls = compute_periodogram(time, mag, err, fmin=1/timespan, fmax=1.0, fres=fres)
            best_idxs = find_local_maxima(ampls, how_many=10) # This come in decreasing order of amplitude
            best_frequencies = freqs[best_idxs]
            best_amplitudes = ampls[best_idxs]
            best = {'best_frequencies': best_frequencies.tolist(), 'best_amplitudes': best_amplitudes.tolist()}
            result.append(simple_features | best)
        elif task is Task.FIT_FOURIER_SERIES_MLE:
            best_frequencies = jnp.asarray(row['best_frequencies'][0].to_numpy())
            n_harmonics = 1
            for freq in best_frequencies:
                params = fit_fourier_series_plus_red_noise(time, mag, err, freq, n_harmonics=n_harmonics)
                result.append({'sourceid': sid, 'frequency': freq, 'n_harmonics': n_harmonics} | params)
        elif task is Task.FIT_RED_NOISE_MLE:
            _, params, mle = fit_gp_red_noise(time, mag, err, fit_gamma=True)
            result.append({'sourceid': sid, 'mle': mle} | params)
        elif task is Task.FIT_RED_NOISE_MCMC:
            samples, r_hats, n_effs = mcmc_red_noise(time, mag, err)
            result.append(
                {'sourceid': row['sourceid'].item()} | r_hats | n_effs | {k: v.tolist() for k, v in samples.items()}
            )
        #elif task is Task.MONTE_CARLO_FAP_MLE:
        #    faps = mc_fap_distribution(time, mag, err, 1./timespan, 1.0, fres, num_reps=1000)
        #    result.append({'sourceid': sid} | faps)
    pl.from_dicts(result).write_parquet(write_path)


if __name__ == '__main__':
    # jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('path_to_dataset', type=str)
    parser.add_argument('save_directory', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('task', type=Task, choices=list(Task), help='Which task to run')
    args = parser.parse_args()
    path_dataset = Path(args.path_to_dataset)
    save_directory = Path(args.save_directory)
    save_directory.mkdir(exist_ok=True)
    parquet_list = list(path_dataset.glob('*.parquet'))
    if len(parquet_list) == 0:
        raise ValueError('Could not find light curve parquets')
    apply_in_parallel(
        partial(extract_from_parquet, save_dir=save_directory, task=args.task),
        parquet_list,
        n_jobs=args.n_jobs,
        description='Extracting features'
    )
