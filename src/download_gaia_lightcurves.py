import logging
import argparse
from pathlib import Path
import polars as pl
from tqdm import tqdm
from astroquery.gaia import Gaia
import numpy as np


logger = logging.getLogger(__name__)


def get_time_series(sourceids: list[int]) -> pl.DataFrame:
    datalink = Gaia.load_data(ids=sourceids,
                              data_release="Gaia DR3",
                              retrieval_type="EPOCH_PHOTOMETRY",
                              data_structure="INDIVIDUAL",
                              verbose=False)
    df_lcs = []
    for sid in sourceids:
        key = f'EPOCH_PHOTOMETRY-Gaia DR3 {sid}.xml'
        df_lcs.append(
            pl.DataFrame(np.ma.filled(datalink[key][0].array)).filter(
                ~pl.col('variability_flag_g_reject')
            ).with_columns(
                pl.lit(sid).alias('sourceid'),
                ((2.5/np.log(10))/pl.col('g_transit_flux_over_error')).alias('g_valerr')
            ).rename(
                {'g_transit_time': 'g_obstimes', 'g_transit_mag': 'g_val'}
            ).group_by('sourceid').agg(['g_obstimes', 'g_val', 'g_valerr'])
        )
    return pl.concat(df_lcs)


def download_light_curves(download_path: Path, sources: pl.DataFrame, batch_size: int = 2000) -> None:
    download_path.mkdir(exist_ok=True, parents=True)
    total_sources = len(sources)
    logger.info(f'Downloading {total_sources} sources')
    
    Gaia.login()
    for idx, frame in tqdm(enumerate(sources.iter_slices(n_rows=batch_size)),
                           total=(total_sources+batch_size-1)//batch_size):
        sids = frame['sourceid'].to_list()
        lc_part_path = Path(download_path / f'part_{idx}.parquet')
        if not lc_part_path.exists():
            lcs = get_time_series(sids)
            lcs.write_parquet(lc_part_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaia light curve downloaders')
    parser.add_argument('data_directory', type=str)
    parser.add_argument('source_selection_parquet', type=str)
    args = parser.parse_args()
    df_sids_path = Path(args.source_selection_parquet)
    download_path = Path(args.data_directory) / df_sids_path.stem
    logger.info(f'Selecting sources from {df_sids_path.name}')
    df_sids = pl.scan_parquet(df_sids_path).select('sourceid').collect()
    download_light_curves(download_path, df_sids)
