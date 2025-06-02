# *Gaia* AGN Survey of Periodic variability (GASP)

This repository holds the implementation of the methods described in ["Periodic Variability in Space Photometry of 181 New Supermassive Black Hole Binary Candidates"](https://arxiv.org/abs/2505.16884). Below you will find instructions to download the data from the *Gaia* archive and process it as described in the paper. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). For convenience, the data and results can be downloaded directly from Zenodo (link pending). 

If you are only interested in the metadata of the candidates please see the `results/gasp.csv` (48k sources) and `results/gasp_only_candidates.csv` (181 sources) in this repository and their electronic versions (link pending). These lists provide *Gaia* source identifiers, the corresponding Bayes factor, positions, and identifiers from crossmatched surveys.


## Instructions

Install the dependencies using `uv` and activate the virtual environment (in bash)

```bash
uv sync
 source .venv/bin/activate
```

Download the light curve of the initial *Gaia* QSO selection: 

```bash
python src/download_gaia_lightcurves.py data/ data/qso_initial_selection.parquet
```

See `notebooks/gaia_qso_initial_selection.ipynb` for the selection criteria that were used.

