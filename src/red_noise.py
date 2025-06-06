import argparse
from pathlib import Path
from functools import partial
import numpy as np
import polars as pl
from preprocessing import pack_light_curve, preprocess_lc
from parallel_utils import apply_in_parallel
from periodicity import extract_dominant_frequency_nifty


def rednoise(seed, time, timescale, varscale):

    """
    INPUT:  . time[0..Ntime-1]: time points
            . timescale[0..Ncomp-1]: time scale tau of each red noise component
            . varscale[0..Ncomp-1]: variation scale of each red noise component
            
    OUTPUT: . signal[0..Ntime-1]: signal containing all red noise components
    
    EXAMPLE: time = np.linspace(0,100,10000)
             signal = rednoise(time, np.array([20.0]), np.array([1.0]))
             
             
    """

    rng = np.random.default_rng(seed)
    Ntime = len(time)
    Ncomp = len(timescale)

    # Set the kick (= excitation) timestep to be one 100th of the
    # shortest noise time scale (i.e. kick often enough).

    kicktimestep = min(timescale) / 100.0

    # Predefine some arrays

    signal = np.zeros(Ntime)
    noise = np.zeros(Ncomp)
    mu = np.zeros(Ncomp)
    sigma = np.sqrt(kicktimestep/timescale)*varscale

    # Warm up the first-order autoregressive process

    for i in range(2000):
        noise = noise * (1.0 - kicktimestep / timescale) + rng.normal(mu, sigma)

    # Start simulating the granulation time series

    delta = 0.0
    currenttime = time[0] - kicktimestep

    for i in range(Ntime):

        # Compute the contribution of each component separately.
        # First advance the time series right *before* the time point i,

        while((currenttime + kicktimestep) < time[i]):
            noise = noise * (1.0 - kicktimestep / timescale) + rng.normal(mu, sigma)
            currenttime = currenttime + kicktimestep

        # Then advance the time series with a small time step right *on* time[i]

        delta = time[i] - currenttime
        noise = noise * (1.0-delta/timescale) + rng.normal(mu, np.sqrt(delta/timescale)*varscale)
        currenttime = time[i]

        # Add the different components to the signal.

        signal[i] = np.sum(noise)
    return signal


def batch_simulator(parquet_path: Path,
                    result_dir: Path,
                    taus: list[float] | np.ndarray,
                    max_lcs: int = 20,
                    n_repetitions: int = 5):
    df_lcs = pl.read_parquet(parquet_path)
    save_path = result_dir / parquet_path.name
    if save_path.exists():
        return None
    if len(df_lcs) > max_lcs:
        df_lcs = df_lcs.sample(max_lcs, seed=1234)
    simulated_df = []
    amplitudescales = np.array([1.0])
    for k in range(len(df_lcs)):
        row = df_lcs.slice(k, 1)
        df_lc = pack_light_curve(row, remove_extreme_errors=True)['g']
        time, _, err = preprocess_lc(df_lc, scale_time=False, center_mag=True, scale_mag=True)
        sid = row['sourceid'].item()
        T = np.amax(time) - np.amin(time)
        for tau in taus:
            timescales = np.array([tau])
            for seed in range(n_repetitions):
                mag = rednoise(seed, time, timescales, amplitudescales)
                mag = mag/np.std(mag)
                mag += np.random.randn(len(err))*err
                period = 1.0/extract_dominant_frequency_nifty(
                    time, mag, err, fmin=1/T, fmax=25, fres=1/(10*T)
                )['NUFFT_frequency']
                simulated_df.append(
                    {'sourceid': sid, 'timescale': tau,
                     'seed': seed, 'period': period, 'gt100d': (period > 100) & (period < T/1.5),
                     'g_obstimes': list(time), 'g_val': list(mag), 'g_valerr': list(err)}
                )
    pl.DataFrame(simulated_df).write_parquet(save_path)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Red noise simulator')
    parser.add_argument('path_to_dataset', type=str)
    parser.add_argument('save_directory', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    path_dataset = Path(args.path_to_dataset)
    save_directory = Path(args.save_directory)
    save_directory.mkdir(exist_ok=True)
    parquet_list = sorted(list(path_dataset.glob('*.parquet')))
    if len(parquet_list) == 0:
        raise ValueError('Could not find light curve parquets')
    apply_in_parallel(
        partial(batch_simulator,
                result_dir=save_directory,
                taus=np.linspace(10, 500, 20)
                ),
        parquet_list,
        n_jobs=args.n_jobs,
        description='Simulating red noise'
    )
