
import argparse
from functools import partial
from urllib.request import urlopen
from pathlib import Path
import polars as pl
import numpy as np

from parallel_utils import apply_in_parallel


BASE = "https://cdn.gea.esac.esa.int/Gaia/gdr3/Photometry/epoch_photometry/"
MD5_NAME = "_MD5SUM.txt"

def read_text_cached(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    txt = urlopen(url).read().decode("utf-8", errors="replace")
    cache_path.write_text(txt, encoding="utf-8")
    return txt

def md5sum_filenames_cached(md5_url: str, cache_path: Path) -> list[str]:
    txt = read_text_cached(md5_url, cache_path)

    files: list[str] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        fname = parts[-1]  # md5sum format: "<md5>  <filename>"
        if fname.startswith("EpochPhotometry_") and fname.endswith(".csv.gz"):
            files.append(fname)

    return files

def to_float_list(name: str) -> pl.Expr:
    return pl.col(name).str.replace_all(r"\bNaN\b", "null").str.json_decode(dtype=pl.List(pl.Float64))
    
def to_bool_list(name: str) -> pl.Expr:
    return pl.col(name).str.json_decode(dtype=pl.List(pl.Utf8)).list.eval(
        pl.when(pl.element().is_null()).then(None).otherwise(pl.element().str.to_lowercase() == "true")
    ).cast(pl.List(pl.Boolean))

def create_save_path(filename: str, save_dir: Path) -> Path:
    stem = filename.removesuffix(".csv.gz")
    return save_dir / f"{stem}.parquet"

def filter_sink_gaia_csv(filename: str, save_dir: Path, df_selection_path: Path) -> str:
    url = BASE + filename
    df_selection = pl.scan_parquet(df_selection_path).select('sourceid')
    try:
        pl.scan_csv(url, comment_prefix='#').select(
            ['source_id', 'g_transit_time', 'g_transit_mag', 'g_transit_flux_over_error', 'variability_flag_g_reject']
        ).rename(
            {'source_id': 'sourceid', 'g_transit_time': 'g_obstimes', 'g_transit_mag': 'g_val'}
        ).join(df_selection, on='sourceid', how='semi').with_columns(
            to_float_list('g_obstimes'),
            to_float_list('g_val'),
            pl.lit(2.5/np.log(10)).truediv(to_float_list('g_transit_flux_over_error')).alias('g_valerr'),
            to_bool_list('variability_flag_g_reject'),
        ).drop('g_transit_flux_over_error').sink_parquet(create_save_path(filename, save_dir))
    except BaseException as e:
        return f"ERROR {filename} {e}"
    return f"OK {filename}"

def redistribute_parquets(
    parquet_dir: Path,
    batch_size: int = 1000,
    pattern: str = "part_{:06d}.parquet",
) -> None:
    parquet_dir = Path(parquet_dir).resolve()
    parquet_files = sorted(parquet_dir.glob("EpochPhotometry_*.parquet"))

    buf: pl.DataFrame | None = None
    shard_idx = 0
    total_written = 0

    for f in parquet_files:
        df = pl.read_parquet(f)
        if df.height == 0:
            continue
        if buf is None:
            buf = df
        else:
            buf = pl.concat([buf, df], how="vertical", rechunk=False)

        while buf.height >= batch_size:
            shard_idx += 1
            out_path = parquet_dir / pattern.format(shard_idx)
            buf.slice(0, batch_size).write_parquet(out_path)
            total_written += batch_size
            buf = buf.slice(batch_size)

    # write remainder 
    if buf is not None and buf.height > 0:
        shard_idx += 1
        out_path = parquet_dir / pattern.format(shard_idx)
        buf.write_parquet(out_path)
        total_written += buf.height
    print(f"Output shards: {shard_idx}")
    print(f"Total light curves written: {total_written}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaia light curve bulk downloader')
    parser.add_argument('data_directory', type=str)
    parser.add_argument('source_selection_parquet', type=str)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()
    df_sids_path = Path(args.source_selection_parquet)
    if not df_sids_path.exists():
        raise FileExistsError(f"File {df_sids_path} does not exist")
    download_path = Path(args.data_directory) / df_sids_path.stem
    download_path.mkdir(exist_ok=True, parents=True)
    print(f"Selecting sources from {df_sids_path.name}")
    files = md5sum_filenames_cached(BASE + MD5_NAME, download_path / MD5_NAME)
    if not args.overwrite:
        files = [f for f in files if not create_save_path(f, download_path).exists()]
    if len(files) > 0:
        apply_in_parallel(
            partial(filter_sink_gaia_csv, save_dir=download_path, df_selection_path=df_sids_path), 
            data_list=files,
            n_jobs=args.n_jobs,
            description="Downloading Gaia time series",
        )
    print("Redistributing light curves in evenly sampled parquets")
    redistribute_parquets(download_path, batch_size=1000)
