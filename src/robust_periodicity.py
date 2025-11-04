from pathlib import Path
import argparse
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp
import jax
from jax import jit, random, vmap
import jaxopt
from tinygp import GaussianProcess, kernels
import polars as pl
from nifty_ls import finufft
from preprocessing import pack_light_curve
from parallel_utils import apply_in_parallel
# from kernels_tinygp import PoweredExp

from jax import numpy as jnp
from tinygp.helpers import JAXArray
from tinygp.kernels import Stationary

class PoweredExp(Stationary):

    gamma: JAXArray | float | None = None

    def __check_init__(self):
        if self.gamma is None:
            raise ValueError("Missing required argument 'gamma'")

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        r = self.distance.distance(X1, X2) / self.scale
        return jnp.exp(-r ** self.gamma)


def build_gp(theta, X, Yerr, kernel_fn, mean_fn=None):
    amps = jnp.exp(theta["log_amps"])
    scales = jnp.exp(theta["log_scales"])
    gamma = 1.0 + 1./(1. + jnp.exp(-theta['logit_gamma']))
    return GaussianProcess(
        amps * kernel_fn(scales, gamma=gamma),
        X, 
        diag=Yerr**2, 
        mean=theta["mean"] if mean_fn is None else partial(mean_fn, theta["mean"])
    )

def find_local_maxima(x: np.ndarray,
                      how_many: int,
                      ) -> np.ndarray:
    local_maxima = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.argsort(x[1:-1][local_maxima])[::-1][:how_many]
    return np.arange(1, len(x)-1)[local_maxima][idx]


def compute_periodogram(time, mag, err, fmin: float, fmax: float, fres: float):
    periodogram = partial(
        finufft.lombscargle,
        t=time, y=mag, dy=err, nthreads=1,
    )
    Nf = int((fmax - fmin) / fres)
    freqs = fmin + np.arange(Nf) * fres
    ampls = periodogram(fmin=fmin, df=fres, Nf=len(freqs))
    return freqs, ampls

def fit_gp(time, mag, err, period: float, n_harmonics: int, init_scale: float = 100.0, init_logit_gamma: float = 0.):
    if n_harmonics > 0:
        initial_mean = np.zeros(shape=(n_harmonics*2 + 1,))
        initial_mean[0] = np.mean(mag)
        mean_fn = partial(fourier_series_mean_function, period=period, n_harmonics=n_harmonics)
    else:
        initial_mean = np.mean(mag)
        mean_fn = None
    theta_init = {
        "mean": initial_mean,
        "log_amps": np.log(np.std(mag)),
        "log_scales": np.log(init_scale),
        "logit_gamma": init_logit_gamma, 

    }
    build_gp_ = partial(build_gp, X=time, Yerr=err, kernel_fn=PoweredExp, mean_fn=mean_fn)
   
    @jax.jit
    def loss(params):
        return -build_gp_(params).log_probability(mag)
        
    solver = jaxopt.ScipyMinimize(fun=loss, jit=True)
    soln = solver.run(theta_init)
    gp = build_gp_(soln.params)
    return soln, gp

def fit_gp_red_noise(time, mag, err, init_scale: float = 100., init_logit_gamma: float = 0.):
    soln, gp = fit_gp(time, mag, err, n_harmonics=0, period=1.0, init_scale=init_scale, init_logit_gamma=init_logit_gamma)
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


def false_alarm_probabilities(freqs, faps, time, mag, err, seed=1234, num_reps=1000):
    gp, params, mle = fit_gp_red_noise(time, mag, err)
    params = {k: v.tolist() for k, v in params.items()}
    params['mle'] = mle
    gp_prior_samples = gp.sample(random.PRNGKey(seed), shape=(num_reps,))
    red_noise_ampl = []
    for gp_prior in gp_prior_samples:
        test_statistic = partial(fit_sine_wave, x=time, y=gp_prior, yerr=err)
        red_noise_ampl.append(vmap(test_statistic)(period=1./freqs))
    red_noise_ampl = jnp.stack(red_noise_ampl)
    return {str(fap): jnp.quantile(red_noise_ampl, q=1-fap, axis=0).tolist() for fap in faps} | params

"""
def process_single_lightcurve(time, mag, err, fmin, fmax, fres, k=100, num_reps=1000, faps=[1e-2, 1e-3, 1e-4]):
    freqs, ampls = compute_periodogram(time, mag, err, fmin, fmax, fres)
    best_idxs = find_local_maxima(ampls, k)
    sort_idx = np.argsort(freqs[best_idxs])
    best_idxs = best_idxs[sort_idx]
    best_frequencies = freqs[best_idxs]
    best_amplitudes = ampls[best_idxs]
    best = {'best_frequencies': best_frequencies.tolist(), 'best_amplitudes': best_amplitudes.tolist()}
    return best | false_alarm_probabilities(best_frequencies, faps, time, mag, err, num_reps=num_reps)
"""

def split_lightcurve_in_two(time, mag, err, how):
    n = len(time)
    n_even = (n // 2) * 2
    match how:
        case 'random':
            perm = np.random.permutation(n)[:n_even]
            i1 = np.sort(perm[: n_even // 2])
            i2 = np.sort(perm[n_even // 2 :])
        case 'half':
            i_all = np.arange(n_even)
            i1 = i_all[: n_even // 2]
            i2 = i_all[n_even // 2 :]
        case 'time':
            mid = 0.5 * (time[0] + time[-1])
            mask = time <= mid
            i1 = np.nonzero(mask)[0]
            i2 = np.nonzero(~mask)[0]
        case _:
            raise ValueError("how must be one of {'random','half','time'}")

    t1, m1, e1 = time[i1], mag[i1], err[i1]
    t2, m2, e2 = time[i2], mag[i2], err[i2]
    return (t1, m1, e1), (t2, m2, e2)

def _wrap_angle(x):
    """Wrap angle to (-pi, pi]."""
    return (x + jnp.pi) % (2*jnp.pi) - jnp.pi

def amp_phase_from_ab(a, b, Vaa, Vbb, Vab):
    A = np.sqrt(a**2 + b**2)
    phi = np.arctan2(b, a)
    if A == 0.0:
        # Degenerate case: no amplitude; phase undefined, set large SEs
        return 0.0, 0.0, 0.0, np.inf
    # Delta-method variances
    var_A = (a*a*Vaa + b*b*Vbb + 2*a*b*Vab) / (A*A)
    var_phi = (b*b*Vaa + a*a*Vbb - 2*a*b*Vab) / (A**4)
    var_A = float(max(var_A, 0.0))
    var_phi = float(max(var_phi, 0.0))
    sigma_A = np.sqrt(var_A)
    sigma_phi = np.sqrt(var_phi)
    return A, phi, sigma_A, sigma_phi


def stability_test(time, mag, err, frequency, debug: bool = False):
    (t1, m1, e1), (t2, m2, e2) = split_lightcurve_in_two(time - np.mean(time), mag, err, 'time')
    soln, _ = fit_gp(jnp.asarray(t1), jnp.asarray(m1), jnp.asarray(e1), period=1./frequency, n_harmonics=1)
    params, cov_params = soln.params['mean'], soln.state.hess_inv[-3:, -3:]
    #_, params, cov_params = fit_sine_wave(jnp.asarray(t1), jnp.asarray(m1), jnp.asarray(e1), 1/best_freq, return_params=True)
    A1, phi1, sigma_A1, sigma_phi1 = amp_phase_from_ab(params[1], params[2], cov_params[1, 1], cov_params[2, 2], cov_params[2, 1])
    soln, _ = fit_gp(jnp.asarray(t2), jnp.asarray(m2), jnp.asarray(e2), period=1./frequency, n_harmonics=1)
    params, cov_params = soln.params['mean'], soln.state.hess_inv[-3:, -3:]
    #_, params, cov_params = fit_sine_wave(jnp.asarray(t2), jnp.asarray(m2), jnp.asarray(e2), 1/best_freq, return_params=True)
    A2, phi2, sigma_A2, sigma_phi2 = amp_phase_from_ab(params[1], params[2], cov_params[1, 1], cov_params[2, 2], cov_params[2, 1])
    delta_A = np.abs(A1 - A2)
    delta_phi_raw = phi1 - phi2
    delta_phi = _wrap_angle(delta_phi_raw)
    sigma_delta_A = np.sqrt(sigma_A1**2 + sigma_A2**2)
    sigma_delta_phi = np.sqrt(sigma_phi1**2 + sigma_phi2**2)
    z_delta_A = delta_A/sigma_delta_A
    z_delta_phi = delta_phi/sigma_delta_phi
    snr1 = A1/sigma_A1
    snr2 = A2/sigma_A2
    if debug:
        _, ax = plt.subplots(figsize=(6, 3))
        mid = (time[-1] + time[0])/2
        t_ = np.linspace(np.amin(t1), mid, 100)
        ax.plot(t_, A1*np.cos(2*np.pi*t_*frequency - phi1))
        t_ = np.linspace(mid, np.amax(t2), 100)
        ax.plot(t_, A2*np.cos(2*np.pi*t_*frequency - phi2))
        ax.errorbar(time, mag-np.mean(mag), err, fmt='.', c='k')
        ax.invert_yaxis()
    return snr1, snr2, z_delta_A, z_delta_phi, np.abs(delta_phi)

def stability_test_f(time, mag, err, frequencies):
    keys = ("snr1", "snr2", "z_delta_A", "z_delta_phi", "delta_phi")
    rows = [stability_test(time, mag, err, f) for f in frequencies]
    if not rows:
        return {k: [] for k in keys}
    cols = zip(*rows)
    return {k: list(col) for k, col in zip(keys, cols)}

from enum import StrEnum

class Task(StrEnum):
    PERIODOGRAM_MAXIMA = "periodogram_maxima"
    FALSE_ALARM_PROBABILITIES = "false_alarm_probabilities"
    STABILITY_METRICS = "stability_metrics"

def extract_from_parquet(parquet_path: Path,
                         save_dir: Path,
                         task: Task,
                         overwrite: bool = False) -> None:
    (save_dir / task.value).mkdir(exist_ok=True, parents=True)
    write_path = save_dir / task.value / parquet_path.name
    if not overwrite and (write_path).exists():
        return None
    df = pl.read_parquet(
        parquet_path,
        columns=['sourceid', 'g_obstimes', 'g_val', 'g_valerr']
    )
    if df.height == 0:
        return None
    if task is Task.FALSE_ALARM_PROBABILITIES or task is Task.STABILITY_METRICS:
        # I need the best frequencies from PERIODOGRAM_MAXIMA
        df = df.join(
            pl.read_parquet(
                save_dir / 'periodogram_maxima' / parquet_path.name,
                columns=['sourceid', 'best_frequencies']
            ),
            on='sourceid'
        )
    result = []
    for k in range(df.height):
        row = df.slice(k, 1)
        lc = pack_light_curve(row, remove_extreme_errors=True)
        time, mag, err = lc['g']
        sid = row['sourceid'][0]
        if task is Task.PERIODOGRAM_MAXIMA:
            simple_features = {
                'sourceid': sid,
                'magnitude_mean': np.mean(mag),
                'magnitude_std': np.std(mag),
                'time_duration': time[-1] - time[0]
                }
            #freqs, ampls = compute_periodogram(time, mag, err, fmin=7e-4, fmax=25.0, fres=1e-5)
            freqs, ampls = compute_periodogram(time, mag, err, fmin=7e-4, fmax=1.0, fres=1e-5)
            best_idxs = find_local_maxima(ampls, how_many=10) # This come in decreasing order of amplitude
            best_frequencies = freqs[best_idxs]
            best_amplitudes = ampls[best_idxs]
            best = {'best_frequencies': best_frequencies.tolist(), 'best_amplitudes': best_amplitudes.tolist()}
            result.append(simple_features | best)
        elif task is Task.FALSE_ALARM_PROBABILITIES:
            best_frequencies = jnp.asarray(row['best_frequencies'][0].to_numpy())
            faps = false_alarm_probabilities(best_frequencies, [1e-3, 1e-4], time, mag, err, num_reps=200)
            result.append({'sourceid': sid} | faps)
        elif task is Task.STABILITY_METRICS:
            best_frequencies = jnp.asarray(row['best_frequencies'][0].to_numpy())
            try:
                m = stability_test_f(time, mag, err, best_frequencies)
                result.append({'sourceid': sid} | m)
            except Exception as e:
                print(f"Source {sid} failed due to {e}")
                

    pl.from_dicts(result).write_parquet(write_path)


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
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
