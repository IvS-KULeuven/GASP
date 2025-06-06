# *Gaia* AGN Survey of Periodic variability (GASP)

This repository holds the implementation of the methods described in ["Periodic Variability in Space Photometry of 181 New Supermassive Black Hole Binary Candidates"](https://arxiv.org/abs/2505.16884). 

For the metadata of the candidates, see the `results/gasp.csv` (48k sources) and `results/gasp_only_candidates.csv` (181 sources) in this repository and their electronic versions (link pending). These lists provide *Gaia* source identifiers, the corresponding Bayes factor, positions, and identifiers from crossmatched surveys. The light curves of the 181 candidates can be found in `data/gaia_smbhb_candidates_lc_compressed.parquet`.

The contents of the `notebooks` directory in this repository include:

- `gaia_qso_initial_selection.ipynb`: Shows the criteria used to create the initial *Gaia* AGN selection. The result of this notebook is `gaia_qso_initial_selection.parquet`.
- `gaia_qso_periods.ipynb`: Details how the Lomb-Scargle periods are used to filter the initial selection. The result of this notebook is the 48k sources found in `data/gaia_qso_period_gt100d/`. (*) This notebook assumes that the light curves from the initial selection are in `data/gaia_qso_initial_selection` (see note below).
- `gaia_MCMC_example.ipynb`: Demonstrates how to fit the Gaussian processes and estimate the Bayesian evidences with a few *Gaia* sources. 
- `gaia_MCMC_population.ipynb`: Shows how the Bayes factors are computed from the evidences and how the final selection of SMBHB candidates is produced. The results of this notebook are `results/bayes_factors.parquet` and the plots in `notebooks/figures`. (*) This notebook assummes that the MCMC traces and evidences exist in `results/mcmc` (see note below).
- `gasp_crossmatch.ipynb`: Enriches the GASP catalogue with metadata and source identifiers from other AGN catalogues. It also explores the *Gaia* time series of SMBHB candidates from the literature. The result of this notebook is `results/gasp.csv`.

**Note:** To fully execute the notebooks marked with (*) you can either download the data and results from Zenodo or follow the instructions below to download the data from the *Gaia* archive and process it as described in the paper.

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). After installing `uv` you can setup and active the Python virtual environment using

```bash
uv sync
source .venv/bin/activate
```

The above command should take no more than a few minutes. If you use a non-bash shell, you may need to change the above command with appropiate activation script in `.venv/bin`.

Note: The implementation found in this repository have only been tested on Linux operating systems.

## Reproduction instructions

(1) Download the light curves of the initial *Gaia* QSO selection 

```bash
python src/download_gaia_lightcurves.py data/ data/gaia_qso_initial_selection.parquet
```

(2) Estimate the dominant period of the *G* band time series:

```bash
python src/periodicity.py data/gaia_qso_initial_selection/ results/periods 8
```

Set the number of threads (8 in the example above) depending on your hardware. 


(3) Run MCMC on the selected *Gaia* sources for each of the Gaussian process (GP) models that you wish to compare, in this case:

```bash
python src/mcmc.py data/gaia_qso_period_gt100d/ results/mcmc cos 8
python src/mcmc.py data/gaia_qso_period_gt100d/ results/mcmc red 8
```


(4) Finally compute the Bayesian evidence for each of the selected GP models with

```bash
python src/evidence.py results/mcmc/cos/ 8
python src/evidence.py results/mcmc/red/ 8
```


