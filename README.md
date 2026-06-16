# *Gaia* AGN Survey of Periodic variability (GASP)

This repository holds the implementation of the methods described in ["A search for periodic AGN variability in Gaia Data Release 3"](https://arxiv.org/abs/2505.16884). 

The contents of the `notebooks` directory in this repository include:

- `gaia_qso_initial_selection.ipynb`: Shows the criteria used to create the initial *Gaia* AGN selection. 
- `methodology_example.ipynb`: Shows all the steps of the pipeline interactively for a single source

The dominant frequencies, red noise parameters and p-values can be found in the `result` folder of this repository.

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). After installing `uv` you can setup and active the Python virtual environment using

```bash
uv sync
```

The above command should take no more than a few minutes. 

Note: The implementation found in this repository have only been tested on Linux operating systems.

## Reproduction instructions

(1) Download the light curves of the initial *Gaia* AGN selection 

```bash
uv run python src/gaia_bulk_download.py --n_jobs 4 data/ data/gaia_qso_initial_selection_40obs_500d.parquet
```


(2) Estimate the dominant period of the *G* band time series:

```bash
uv run python src/robust_periodicity.py data/gaia_qso_initial_selection_40obs_500d results 8 periodogram_maxima
```

Set the number of threads (8 in the example above) depending on your hardware. 


(3) Fit the damped random walk model using maximum a posteori with Laplace-approximation uncertainties:

```bash
uv run python src/robust_periodicity.py data/gaia_qso_initial_selection_40obs_500d results 8 red_noise_laplace
```


(4) Draw DRW realizations from the posteriors and save the false alarm probabilities

```bash
uv run python src/robust_periodicity.py data/gaia_qso_initial_selection_40obs_500d results 8 pval_mc_laplace
```


