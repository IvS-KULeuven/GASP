# *Gaia* AGN Survey of Periodic variability (GASP)

This repository holds the implementation of the methods described in ["Periodic Variability in Space Photometry of 181 New Supermassive Black Hole Binary Candidates"](https://arxiv.org/abs/2505.16884). Below you will find instructions to download the data from the *Gaia* archive and process it as described in the paper. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). For convenience, the data and MCMC results can be downloaded directly from Zenodo (link pending). 

If you are only interested in the metadata of the candidates please see the `results/gasp.csv` (48k sources) and `results/gasp_only_candidates.csv` (181 sources) in this repository and their electronic versions (link pending). These lists provide *Gaia* source identifiers, the corresponding Bayes factor, positions, and identifiers from crossmatched surveys.

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Install the dependencies and activate the virtual environment (in bash) using

```bash
uv sync
source .venv/bin/activate
```

## Reproduction instructions

(1) Download the light curves of the initial *Gaia* QSO selection 

```bash
python src/download_gaia_lightcurves.py data/ data/gaia_qso_initial_selection.parquet
```

See `notebooks/gaia_qso_initial_selection.ipynb` for the selection criteria that were used.

(2) Estimate the dominant period of the *G* band time series:

```bash
python src/periodicity.py data/gaia_qso_initial_selection/ results/periods 8
```

Set the number of threads (8 in the example above) depending on your hardware. 

Please see `notebooks/gaia_qso_periods.ipynb` for details on how the estimated periods are used to filter the initial selection and produce the input for the next step.

(3) Run MCMC on the selected *Gaia* sources for each of the Gaussian process (GP) models that you wish to compare, in this case:

```bash
python src/mcmc.py data/gaia_qso_period_gt100d/ results/mcmc cos 8
python src/mcmc.py data/gaia_qso_period_gt100d/ results/mcmc red 8
```

To explore the results visually for a few *Gaia* sources, see: `notebooks/gaia_MCMC_example.ipynb`. 

(4) Finally compute the Bayesian evidence for each of the selected GP models with

```bash
python src/evidence.py results/mcmc/cos/ 8
python src/evidence.py results/mcmc/red/ 8
```

See `notebooks/gaia_MCMC_population.ipynb` for details on how the Bayes factor are computed from the evidence and how the final selection of SMBHB is produced.

