import numpy as np
import polars as pl


def pack_light_curve(df_row: pl.DataFrame,
                     remove_extreme_errors: bool = False,
                     use_robust_statistics: bool = True,
                     max_length: int | None = None,
                     ) -> dict[str, np.ndarray]:
    def col_to_array(col: pl.Series) -> np.ndarray | None:
        if col[0] is None:
            return None
        return col[0].to_numpy().astype('float64')
    light_curve = {}
    bands = [col.split('_')[0] for col in df_row.columns if 'obstimes' in col]
    for band in bands:
        time = col_to_array(df_row[f'{band}_obstimes'])
        val = col_to_array(df_row[f'{band}_val'])
        valerr = col_to_array(df_row[f'{band}_valerr'])
        if valerr is None:
            valerr = 0.1*np.ones_like(val)
            remove_extreme_errors = False
        lcb = np.stack([time, val, valerr])
        mask = ~np.isinf(valerr) & ~np.isnan(valerr)
        lcb = lcb[:, mask]
        if remove_extreme_errors:
            valerr = lcb[-1]
            if not use_robust_statistics:
                center = np.mean(valerr)
                scale = np.std(valerr)
            else:
                center = np.median(valerr)
                scale = 1.4826*np.median(np.abs(valerr - center))
            if scale > 0.0:
                inliers = valerr < center + 3*scale
                lcb = lcb[:, inliers]
        if max_length is not None:
            lc_length = lcb.shape[-1]
            if lc_length > max_length:
                idx = np.random.permutation(lcb.shape[-1])[:max_length]
                lcb = lcb[:, idx]
        idx = lcb[0].argsort()
        light_curve[band] = lcb[:, idx]
    return light_curve


def preprocess_lc(lc,
                  scale_time: bool = True,
                  center_mag: bool = True,
                  scale_mag: bool = True,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time, mag, err = lc
    if scale_time:
        time = (time - np.mean(time))/1038
    if center_mag:
        mag = mag - np.mean(mag)
    if scale_mag:
        scale = np.std(mag)
        mag = mag/scale
        err = err/scale
    return time, mag, err


def create_weights_mask(time1: np.ndarray,
                        time2: np.ndarray,
                        window: str,
                        T: float) -> np.ndarray:
    time_diffs = np.abs(time1[:, np.newaxis] - time2[np.newaxis, :])
    match window:
        case 'hann':
            weights = np.cos(np.pi * time_diffs / T)
        case 'rect':
            weights = np.ones_like(time_diffs)
        case _:
            raise ValueError(f"Unrecognized window {window}")
    weights[time_diffs > 0.5 * T] = 0
    return weights/np.sum(weights, axis=1, keepdims=True)


def vectorized_smooth(time: np.ndarray,
                      flux: np.ndarray,
                      window: str = 'hann',
                      T: float = 5.0,
                      chunk_size: int = 1000) -> np.ndarray:
    n = len(time)
    if n <= chunk_size:
        weights = create_weights_mask(time, time, window, T)
        return np.sum(weights * flux, axis=1)
    else:
        fs = 1.0/np.mean(time[1:] - time[:-1])
        max_window_length = int(fs*T)
        new_flux = np.zeros_like(flux)
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            time_chunk = time[chunk_start:chunk_end]
            # Outside this range the window will zero
            window_start = max(chunk_start - max_window_length, 0)
            window_end = min(chunk_end + max_window_length, n)
            time_window = time[window_start:window_end]
            flux_window = flux[window_start:window_end]
            weights = create_weights_mask(time_chunk, time_window, window, T)
            new_flux[chunk_start:chunk_end] = np.sum(weights*flux_window, axis=1)
        return new_flux
