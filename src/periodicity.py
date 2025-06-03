from pathlib import Path
import argparse
from functools import partial
import numpy as np
import polars as pl
from astropy.timeseries import LombScargle
from nifty_ls import finufft
from preprocessing import pack_light_curve
from parallel_utils import apply_in_parallel


def find_local_maxima(x: np.ndarray,
                      how_many: int,
                      ) -> np.ndarray:
    local_maxima = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    idx = np.argsort(x[1:-1][local_maxima])[::-1][:how_many]
    return np.arange(1, len(x)-1)[local_maxima][idx]


def extract_dominant_frequency_nifty(time: np.ndarray,
                                     mag: np.ndarray,
                                     err: np.ndarray,
                                     fmin: float,
                                     fmax: float,
                                     fres: float,
                                     n_best: int = 10,
                                     nthreads: int = 1
                                     ) -> dict[str, float]:
    periodogram = partial(
        finufft.lombscargle,
        t=time, y=mag, dy=err, nthreads=nthreads,
    )

    def get_frequency(k: int, fmin: float, fres: float) -> float:
        return fmin + k*fres
    ampl = periodogram(fmin=fmin, df=fres, Nf=int((fmax-fmin)/fres))
    best_idxs = find_local_maxima(ampl, how_many=n_best)
    best_freq = get_frequency(best_idxs[0], fmin, fres)
    best_ampl = ampl[best_idxs[0]]
    fres_fine = 0.1*fres
    for best_idx in best_idxs:
        fmin_fine = get_frequency(best_idx, fmin, fres) - fres
        fmax_fine = get_frequency(best_idx, fmin, fres) + fres
        fine_ampl = periodogram(
            fmin=fmin_fine,
            df=fres_fine,
            Nf=int((fmax_fine-fmin_fine)/fres_fine)
        )
        max_cand = np.argmax(fine_ampl)
        if fine_ampl[max_cand] > best_ampl:
            best_ampl = fine_ampl[max_cand]
            best_freq = get_frequency(int(max_cand), fmin_fine, fres_fine)
    ls = LombScargle(time, mag, err, normalization='standard')
    fap = ls.false_alarm_probability(
        best_ampl,
        method='baluev',
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=100
    )
    return {'NUFFT_frequency': best_freq, 'NUFFT_amplitude': best_ampl, 'FAP': fap}


def extract_from_parquet(parquet_path: Path,
                         save_dir: Path,
                         overwrite: bool = False) -> None:
    write_path = save_dir / parquet_path.name
    if not overwrite and (write_path).exists():
        return None
    df = pl.read_parquet(
        parquet_path,
        columns=['sourceid', 'g_obstimes', 'g_val', 'g_valerr']
    )
    result = []
    for k in range(df.shape[0]):
        row = df.slice(k, 1)
        lc = pack_light_curve(row, remove_extreme_errors=True)
        time, mag, error = lc['g']
        features = {
            'sourceid': row['sourceid'][0],
            'magnitude_mean': np.mean(mag),
            'magnitude_std': np.std(mag),
            'time_duration': time[-1] - time[0]
            }
        per_features = extract_dominant_frequency_nifty(
            time, mag, error,
            fmin=7e-4, fmax=25.0, fres=1e-5,
        )
        features = features | per_features
        result.append(features)
    pl.from_dicts(result).write_parquet(write_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features')
    parser.add_argument('path_to_dataset', type=str)
    parser.add_argument('save_directory', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    path_dataset = Path(args.path_to_dataset)
    save_directory = Path(args.save_directory)
    save_directory.mkdir(exist_ok=True)
    parquet_list = list(path_dataset.glob('*.parquet'))
    if len(parquet_list) == 0:
        raise ValueError('Could not find light curve parquets')
    apply_in_parallel(
        partial(extract_from_parquet, save_dir=save_directory),
        parquet_list,
        n_jobs=args.n_jobs,
        description='Extracting features'
    )
