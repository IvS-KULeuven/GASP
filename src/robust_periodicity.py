from pathlib import Path
import argparse
from functools import partial
import numpy as np
from parallel_utils import jax_joblib_configuration
jax_joblib_configuration()

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
from astropy.timeseries import LombScargle

from vi import sample_posterior_vi, sample_posterior_laplace, fit_vi

def unpack_light_curve(
        df_row: pl.DataFrame, 
        apply_variability_reject_mask: bool = True
) -> dict[str, np.ndarray]:
    def col_to_array(col: pl.Series) -> np.ndarray:
        return col[0].to_numpy()
    light_curve = {}
    bands = [col.split('_')[0] for col in df_row.columns if 'obstimes' in col]
    for band in bands:
        time = col_to_array(df_row[f'{band}_obstimes'])
        val = col_to_array(df_row[f'{band}_val'])
        valerr = col_to_array(df_row[f'{band}_valerr']) 
        lc_stack = np.stack([time, val, valerr])
        if apply_variability_reject_mask:
            reject_mask = col_to_array(df_row[f'variability_flag_{band}_reject']) 
            lc_stack = lc_stack[:, ~reject_mask]
        light_curve[band] = lc_stack
    return light_curve

def clean_light_curve(
        lc, 
        remove_extreme_errors: bool = True, 
        remove_extreme_values: bool = True, 
        extreme_error_threshold: float = 3.0,
        extreme_value_threshold: float = 1.5) -> dict[str, np.ndarray]:
    clean_light_curve = {}
    for band, lcb in lc.items():
        valerr = lcb[-1]
        mask = ~np.isinf(valerr) & ~np.isnan(valerr)
        lcb = lcb[:, mask]
        if remove_extreme_errors:
            valerr = lcb[-1]
            outlier_abs = valerr > 0.5
            q3, q1 = np.percentile(valerr, (75, 25))
            iqr_err = q3 - q1
            outlier_rel = valerr  > q3 + extreme_error_threshold*iqr_err
            outlier = outlier_abs | outlier_rel
            lcb = lcb[:, ~outlier]
        if remove_extreme_values:
            val = lcb[-2]
            q3, q1 = np.percentile(val, (75, 25))
            iqr_val = q3 - q1
            outlier = (val > q3 + extreme_value_threshold*iqr_val) | (val < q1 - extreme_value_threshold*iqr_val)
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

def build_gp(theta, X, Yerr, mean_fn=None, kernel_fn=None):
    return _build_gp(
        X=X, Yerr=Yerr, log_sigma=theta['log_sigma'], log_tau=theta['log_tau'], mean=theta['mean'],
        mean_fn=mean_fn, logit_gamma=theta['logit_gamma'] if 'logit_gamma' in theta else None, kernel_fn=kernel_fn,
    )


def _build_gp(X, Yerr, log_sigma, log_tau, mean, logit_gamma=None, mean_fn=None, kernel_fn=None):
    sigma = jnp.exp(log_sigma)
    tau = jnp.exp(log_tau)
    if logit_gamma is not None:
        gamma = box_constraint(logit_gamma, 1.0, 2.0)
        kernel = PoweredExp(tau, gamma=gamma)
    else:
        if kernel_fn is None:
            kernel = kernels.quasisep.Exp(tau)
        else:
            kernel = kernel_fn(tau)
    if mean_fn is not None:
        mean = partial(mean_fn, mean)
    return GaussianProcess(sigma**2 * kernel, X, diag=Yerr**2, mean=mean)


def find_local_maxima(x: np.ndarray,
                      how_many: int,
                      ) -> np.ndarray:
    local_maxima = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.argsort(x[1:-1][local_maxima])[::-1][:how_many]
    return np.arange(1, len(x)-1)[local_maxima][idx]

def compute_frequency_grid(fmin: float, fmax: float, fres: float) -> np.ndarray:
    Nf = int((fmax - fmin) / fres)
    return fmin + np.arange(Nf) * fres

def compute_periodogram(
        time: np.ndarray, mag: np.ndarray, err: np.ndarray, 
        fmin: float, fmax: float, fres: float, normalization: str='standard'
) -> np.ndarray:
    periodogram = partial(
        finufft.lombscargle,
        t=time, y=mag, dy=err, nthreads=1, normalization=normalization,
    )
    Nf = int((fmax - fmin) / fres)
    return periodogram(fmin=fmin, df=fres, Nf=Nf)

def max_periodogram_amplitude_distribution(time, vals, err, fmin, fmax, fres, batch_size: int = 2000):
    num_samples = vals.shape[0]
    best_ampl = np.empty(num_samples, dtype=time.dtype)
    best_freq = np.empty(num_samples, dtype=time.dtype)
    freqs = compute_frequency_grid(fmin, fmax, fres)
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        vals_batch = vals[start:stop]
        ampls = compute_periodogram(
            time, np.asarray(vals_batch), np.tile(err[None, :], (vals_batch.shape[0], 1)), 
            fmin, fmax, fres
        )
        idx_max = np.argmax(ampls, axis=-1)
        idx = np.arange(stop-start)
        best_ampl[start:stop] = ampls[idx, idx_max]
        best_freq[start:stop] = freqs[idx_max]
    return best_ampl, best_freq

def median_abs_pairwise(time):
    time = jnp.asarray(time)
    n = time.shape[0]
    diffs = jnp.abs(time[:, None] - time[None, :])
    iu = jnp.triu_indices(n, k=1)
    return jnp.median(diffs[iu])

def fit_gp(time, mag, err, period: float, n_harmonics: int, init_scale: float | None = None, init_logit_gamma: float = 0., fit_gamma: bool = True):
    if n_harmonics > 0:
        initial_mean = jnp.zeros(shape=(n_harmonics*2 + 1,))
        initial_mean.at[0].set(jnp.mean(mag))
        # initial_mean[0] = jnp.mean(mag)
        mean_fn = partial(fourier_series_mean_function, period=period, n_harmonics=n_harmonics)
    else:
        initial_mean = jnp.mean(mag)
        mean_fn = None
    if init_scale is None:
        init_scale = median_abs_pairwise(time)
    theta_init = {
        "mean": initial_mean,
        "log_sigma": jnp.log(jnp.std(mag)),
        "log_tau": jnp.log(init_scale), 
    }
    if fit_gamma:
        theta_init['logit_gamma'] = init_logit_gamma
    build_gp_ = partial(build_gp, X=time, Yerr=err, mean_fn=mean_fn)
   
    @jax.jit
    def loss(params):
        return -build_gp_(params).log_probability(mag)
        
    solver = jaxopt.ScipyMinimize(fun=loss, jit=True)
    soln = solver.run(theta_init)
    gp = build_gp_(soln.params)
    return soln, gp

def fit_gp_red_noise(time, mag, err, init_scale: float | None = None, init_logit_gamma: float = 0., fit_gamma=True):
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

def bayesian_red_noise(x, yerr, y, x_interp=None, fit_gamma: bool = True, truncate_tau: bool = False, kernel_fn = None):
    mu_logtau, sig_logtau = -0.6 + jnp.log(365.5), 1.15 # From Vaughan et al. 2016
    mean = numpyro.sample("mean", dist.Normal(jnp.mean(y), 1.0))
    # log_sigma = numpyro.sample("log_sigma", dist.Normal(jnp.log(jnp.std(y)), 1.0))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(jnp.log(0.1), 1.15))
    if truncate_tau:
        max_tau = 2.*(x[-1] - x[0])
        min_tau = 0.5*jnp.mean(x[1:] - x[:-1])
        log_tau = numpyro.sample("log_tau", dist.TruncatedNormal(mu_logtau, sig_logtau, low=jnp.log(min_tau), high=jnp.log(max_tau)))
    else:
        log_tau = numpyro.sample("log_tau", dist.Normal(mu_logtau, sig_logtau))
    logit_gamma = None
    if fit_gamma:
        logit_gamma = numpyro.sample("logit_gamma", dist.Normal(0, 1))
    gp = _build_gp(X=x, Yerr=yerr, log_sigma=log_sigma, log_tau=log_tau, mean=mean, logit_gamma=logit_gamma, kernel_fn=kernel_fn)
    numpyro.sample("gp", gp.numpyro_dist(), obs=y)
    if y is not None and x_interp is not None:
        numpyro.deterministic("pred", gp.condition(y, x_interp).gp.loc)

def mcmc_red_noise(model, rng_key, time, mag, err,
                   num_warmup: int = 500, num_samples: int = 2500, num_chains: int = 4, progress_bar: bool = False):
    nuts_kernel = NUTS(model, target_accept_prob=0.9, dense_mass=True, max_tree_depth=(5, 10))
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar, jit_model_args=True, chain_method='sequential')
    mcmc.run(rng_key, x=jnp.asarray(time), y=jnp.asarray(mag), yerr=jnp.asarray(err))
    samples = mcmc.get_samples()
    r_hats, n_effs = {}, {}
    for k, v in summary(mcmc.get_samples(group_by_chain=True)).items():
        r_hats[f'r_hat_{k}'] = v['r_hat'].item()
        n_effs[f'n_eff_{k}'] = v['n_eff'].item()
    return samples, r_hats, n_effs

def fit_fourier_series_plus_red_noise(time, mag, err, freq, init_scale: float = 100., init_logit_gamma: float = 0., n_harmonics:int = 1):
    soln, _ = fit_gp(
        jnp.asarray(time), jnp.asarray(mag), jnp.asarray(err), 
        period=1./freq, n_harmonics=n_harmonics, init_scale=init_scale, init_logit_gamma=init_logit_gamma
    )
    params = {k: v.tolist() for k, v in soln.params.items()}
    params['mle'] = soln.state.fun_val
    return params

def top_peaks_over_bins(time, mag, err, freqs: np.ndarray, ampls: np.ndarray, samples_per_peak: float):
    T = time[-1] - time[0]
    fmin, fmax = freqs[0], freqs[-1]
    edges = np.asarray([1.0 / T, 1.5 / T, 3.0 / T, 5.0 / T, 0.01])
    per_bin = {'top_frequency': [], 'top_amplitude': [], 'top_fap': [], 'bin_freq_lo': [], 'bin_freq_hi': [], 'bin_num_freqs': []}
    best_idx = np.argmax(ampls)
    per_bin['top_frequency'].append(freqs[best_idx])
    per_bin['top_amplitude'].append(ampls[best_idx])
    ls = LombScargle(time, mag, err, normalization='standard')
    per_bin['top_fap'].append(ls.false_alarm_probability(per_bin['top_amplitude'], method='baluev', minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak).item())
    per_bin['bin_freq_lo'].append(fmin)
    per_bin['bin_freq_hi'].append(fmax)
    per_bin['bin_num_freqs'].append(len(freqs))
    for _, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m_bin = (freqs >= lo) & (freqs <= hi)
        per_bin['bin_freq_lo'].append(lo)
        per_bin['bin_freq_hi'].append(hi)
        per_bin['bin_num_freqs'].append(np.sum(m_bin))
        if np.any(m_bin):
            idx_rel = np.argmax(ampls[m_bin])
            idx = np.flatnonzero(m_bin)[idx_rel]
            per_bin['top_frequency'].append(freqs[idx])
            per_bin['top_amplitude'].append(ampls[idx])
            per_bin['top_fap'].append(ls.false_alarm_probability(ampls[idx], method='baluev', minimum_frequency=lo, maximum_frequency=hi, samples_per_peak=samples_per_peak).item())
        else:
            per_bin['top_frequency'].append(None)
            per_bin['top_amplitude'].append(None)
            per_bin['top_fap'].append(None)
    return per_bin

def smoothed_mean(counts, n_trial):
    # Beta (1/2, 1/2)
    return (counts + 0.5)/(n_trial + 1)

def batched_p_value_max_periodogram(ampl_test, edges, time, vals, err, fres, batch_size: int = 2000, report_smoothed_mean: bool = True):
    """
    ampl_test and edges are of the same size
    """
    max_samples = vals.shape[0]
    num_samples = 0
    p_values = np.zeros(len(ampl_test), dtype=time.dtype)
    valid_bands = np.ones(len(ampl_test), dtype=bool)
    fmin, fmax = edges[0]
    freqs = compute_frequency_grid(fmin, fmax, fres)
    band_masks = []
    for k, (flo, fhi) in enumerate(edges):
        m_bin = (freqs >= flo) & (freqs <= fhi)
        if not np.any(m_bin): # These will not be updated below
            valid_bands[k] = False
            p_values[k] = np.nan
        band_masks.append(m_bin)
    band_masks = np.asarray(band_masks, dtype=bool)   

    while num_samples < max_samples:
        take = min(batch_size, max_samples - num_samples)
        vals_batch = vals[num_samples:num_samples+take, :]
        ampls = compute_periodogram(
            time, np.asarray(vals_batch), np.tile(err[None, :], (vals_batch.shape[0], 1)), 
            fmin, fmax, fres
        )
        num_samples += take
        for k, (fmask, ampl_star) in enumerate(zip(band_masks, ampl_test)):
            if valid_bands[k]:
                max_in_band = np.amax(ampls[:, fmask], axis=-1)
                p_values[k] += np.sum(max_in_band > ampl_star)
        if num_samples > 10_000:
            if np.all(p_values[valid_bands] > 1e-3*max_samples):
                break
    if report_smoothed_mean:
        p_values = smoothed_mean(p_values, num_samples)
    else:
        p_values /= num_samples
    return p_values, num_samples

from enum import StrEnum

class Task(StrEnum):
    PERIODOGRAM_MAXIMA = "periodogram_maxima"
    FIT_FOURIER_SERIES_MLE = "fourier_mle"
    FIT_RED_NOISE_MLE = "red_noise_mle"
    FIT_RED_NOISE_MAP = "red_noise_map"
    FIT_RED_NOISE_LAPLACE = "red_noise_laplace"
    FIT_RED_NOISE_VI = "red_noise_vi"
    FIT_RED_NOISE_MCMC = "red_noise_mcmc"
    FIT_RED_NOISE_MCMC_GAMMA = "red_noise_mcmcgamma"
    PVAL_MC_MLE = "pval_mc_mle"
    PVAL_MC_MAP = "pval_mc_map"
    PVAL_MC_LAPLACE = "pval_mc_laplace"
    PVAL_MC_MCMC = "pval_mc_mcmc"
    PVAL_MC_MCMC_GAMMA = "pval_mc_mcmcgamma"

def create_save_path(parquet_input_path: Path , task: Task, save_dir: Path) -> Path:
    return save_dir / task.value / parquet_input_path.name

def sample_gp_prior(time, err, log_sigma, log_tau, mean, key, logit_gamma=None):
    return _build_gp(
        X=jnp.asarray(time), Yerr=jnp.asarray(err), log_sigma=log_sigma, log_tau=log_tau, mean=mean, logit_gamma=logit_gamma
    ).sample(key, shape=(1, ))

def extract_from_parquet(parquet_path: Path,
                         save_dir: Path,
                         task: Task,
                         min_cycles: float = 1.0,
                         fmax: float = 25.0,
                         oversampling: float = 100.,
                         random_seed: int = 1234,
                         ) -> None:
    write_path = create_save_path(parquet_path, task, save_dir)
    df = pl.read_parquet(parquet_path)
    if df.height == 0:
        return None

    if 'pval_mc' in task.value:
        gp_task = 'red_noise_'+task.value.split('_')[-1]
        if not (save_dir / gp_task).exists():
            raise ValueError("You need to fit the GPs before estimating the FAPs")
        df = df.join(
            pl.read_parquet(save_dir / gp_task / parquet_path.name), on='sourceid'
        ).join(
            pl.read_parquet(save_dir / Task.PERIODOGRAM_MAXIMA.value / parquet_path.name), on='sourceid'
        )
        model_params = pl.scan_parquet(save_dir / gp_task / parquet_path.name).drop('sourceid').collect().columns
        model_params = [name for name in model_params if 'r_hat' not in name and 'n_eff' not in name]
    if task is Task.FIT_FOURIER_SERIES_MLE:
        for col in ['best_frequencies', 'best_amplitudes']:
            if col not in df.columns:
                raise ValueError(f"Task {task} needs column {col}.")
    result = []
    key = random.PRNGKey(random_seed)
    for k in range(df.height):
        jax.clear_caches()
        row = df.slice(k, 1)
        lc = clean_light_curve(unpack_light_curve(row))
        time, mag, err = lc['g']
        timespan = time[-1] - time[0]
        fmin = min_cycles/timespan
        fres = (1.0 / timespan) / oversampling
        if fmin + fres >= fmax: # May happen if light curve is very short
            continue
        sid = row['sourceid'][0]
        key, subkey = jax.random.split(key) # For VI and MCMC
        if task is Task.PERIODOGRAM_MAXIMA:
            simple_features = {
                'sourceid': sid,
                'num_obs': len(mag),
                'magnitude_mean': np.mean(mag),
                'magnitude_std': np.std(mag),
                'time_duration': timespan
                }
            ampls = compute_periodogram(time, mag, err, fmin=fmin, fmax=fmax, fres=fres)
            freqs = compute_frequency_grid(fmin, fmax, fres)
            best = top_peaks_over_bins(time, mag, err, freqs, ampls, oversampling)
            result.append(simple_features | best)
        elif task is Task.FIT_FOURIER_SERIES_MLE:
            best_frequencies = jnp.asarray(row['best_frequencies'][0].to_numpy())
            n_harmonics = 1
            for freq in best_frequencies:
                params = fit_fourier_series_plus_red_noise(time, mag, err, freq, n_harmonics=n_harmonics)
                result.append({'sourceid': sid, 'frequency': freq, 'n_harmonics': n_harmonics} | params)
        elif task is Task.FIT_RED_NOISE_MLE:
            _, params, mle = fit_gp_red_noise(time, mag, err, fit_gamma=False)
            result.append({'sourceid': sid, 'mle': mle} | params)
        elif task is Task.FIT_RED_NOISE_MAP:
            samples, losses = sample_posterior_vi(
                partial(bayesian_red_noise, fit_gamma=False), rng_key=subkey, guide_type='MAP', num_particles=1, lr=5e-3, max_steps=2_000, early_stopping=True,
                x=jnp.asarray(time), yerr=jnp.asarray(err), y=jnp.asarray(mag)
            )
            result.append({'sourceid': sid, 'elbo': losses[-1]} | {k: v.item() for k, v in samples.items()})
        elif task is Task.FIT_RED_NOISE_VI:
            samples, losses = sample_posterior_vi(
                partial(bayesian_red_noise, fit_gamma=False), rng_key=subkey, guide_type='BNAF',
                x=jnp.asarray(time), yerr=jnp.asarray(err), y=jnp.asarray(mag)
            )
            result.append({'sourceid': sid, 'elbo': losses[-1]} | {k: v.tolist() for k, v in samples.items()})
        elif task is Task.FIT_RED_NOISE_LAPLACE:
            guide, vi_params, losses = fit_vi(
                partial(bayesian_red_noise, fit_gamma=False), rng_key=subkey, guide_type='Laplace', num_particles=1, lr=5e-3,
                x=jnp.asarray(time), yerr=jnp.asarray(err), y=jnp.asarray(mag)
            )
            result.append({'sourceid': sid, 'elbo': losses[-1]} | {'vi_params': vi_params['auto_loc'].tolist()})
        elif task is Task.FIT_RED_NOISE_MCMC:
            samples, r_hats, n_effs = mcmc_red_noise(partial(bayesian_red_noise, fit_gamma=False), subkey, time, mag, err, num_samples=25_000)
            result.append(
                {'sourceid': sid} | r_hats | n_effs | {k: v.tolist() for k, v in samples.items()}
            )
        elif task is Task.FIT_RED_NOISE_MCMC_GAMMA:
            samples, r_hats, n_effs = mcmc_red_noise(partial(bayesian_red_noise, fit_gamma=True), subkey, time, mag, err, num_samples=25_000)
            result.append(
                {'sourceid': sid} | r_hats | n_effs | {k: v.tolist() for k, v in samples.items()}
            )
        #elif task is Task.MONTE_CARLO_FAP_MLE or task is Task.MONTE_CARLO_FAP_MAP:
        #    gp = build_gp(row.to_dicts()[0], X=jnp.asarray(time), Yerr=jnp.asarray(err))
        #    gp_prior_samples = gp.sample(subkey, shape=(10000,))
        #    ampl_dist, _ = max_periodogram_amplitude_distribution(time, gp_prior_samples, err, fmin=fmin, fmax=fmax, fres=fres, batch_size=1000)
        #    result.append({'sourceid': sid, 'max_periodogram_amplitudes': ampl_dist.tolist()})
        elif task is Task.PVAL_MC_MAP:
            gp = build_gp(row.to_dicts()[0], X=jnp.asarray(time), Yerr=jnp.asarray(err))
            gp_prior_samples = gp.sample(subkey, shape=(100_000,)) 
            p_values, num_samples = batched_p_value_max_periodogram(
                row['top_amplitude'][0].to_numpy(), 
                np.stack([row['bin_freq_lo'][0].to_numpy(), row['bin_freq_hi'][0].to_numpy()]).T, 
                time, gp_prior_samples, err, fres, batch_size=1000)
            result.append({'sourceid': sid, 'p_values': p_values.tolist(), 'num_samples': num_samples})
        elif task is Task.PVAL_MC_LAPLACE:
            vi_params = {'auto_loc': jnp.asarray(row.to_dicts()[0]['vi_params'])}
            model = partial(bayesian_red_noise, fit_gamma=False)
            samples = sample_posterior_laplace(
                model, subkey, vi_params, x=jnp.asarray(time), yerr=jnp.asarray(err), y=jnp.asarray(mag), num_samples=100_000
            )
            keys = random.split(subkey, len(samples['log_sigma']))
            gp_prior_samples = jax.jit(jax.vmap(partial(sample_gp_prior, time=time, err=err)))(**samples, key=keys)[:, 0, :]
            p_values, num_samples = batched_p_value_max_periodogram(
                row['top_amplitude'][0].to_numpy(), 
                np.stack([row['bin_freq_lo'][0].to_numpy(), row['bin_freq_hi'][0].to_numpy()]).T, 
                time, gp_prior_samples, err, fres, batch_size=250)
            result.append({'sourceid': sid, 'p_values': p_values.tolist(), 'num_samples': num_samples})
        elif task is Task.PVAL_MC_MCMC or task is Task.PVAL_MC_MCMC_GAMMA:
            keys = random.split(subkey, len(row['log_sigma'][0]))
            samples = {k: jnp.asarray(v) for k, v in row.to_dicts()[0].items() if k in model_params}
            gp_prior_samples = jax.jit(jax.vmap(partial(sample_gp_prior, time=time, err=err)))(**samples, key=keys)[:, 0, :]
            p_values, num_samples = batched_p_value_max_periodogram(
                row['top_amplitude'][0].to_numpy(), 
                np.stack([row['bin_freq_lo'][0].to_numpy(), row['bin_freq_hi'][0].to_numpy()]).T, 
                time, gp_prior_samples, err, fres, batch_size=100)
            result.append({'sourceid': sid, 'p_values': p_values.tolist(), 'num_samples': num_samples})

    pl.from_dicts(result).write_parquet(write_path, compression='zstd', compression_level=22)


if __name__ == '__main__':
    # jax_joblib_configuration()
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('path_to_dataset', type=str)
    parser.add_argument('save_directory', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('task', type=Task, choices=list(Task), help='Which task to run')
    parser.add_argument('--overwrite', type=bool, default=False)
    args = parser.parse_args()
    path_dataset = Path(args.path_to_dataset)
    save_directory = Path(args.save_directory) / path_dataset.name
    task = args.task
    (save_directory / task.value).mkdir(exist_ok=True, parents=True)
    parquet_list = list(path_dataset.glob('part_*.parquet'))
    if len(parquet_list) == 0:
        raise ValueError('Could not find light curve parquets')
    if not args.overwrite:
        parquet_list = [f for f in parquet_list if not create_save_path(f, task, save_directory).exists()]
    apply_in_parallel(
        partial(extract_from_parquet, save_dir=save_directory, task=task, min_cycles=1, fmax=0.01),
        parquet_list,
        n_jobs=args.n_jobs,
        description=f'Executing task {task}'
    )
