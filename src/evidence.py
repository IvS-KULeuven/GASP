import argparse
from pathlib import Path
import polars as pl
import jax
from jax import numpy as jnp
import harmonic as hm

from parallel_utils import apply_in_parallel


def ln_evidence(ln_evidence_inv: float, ln_evidence_inv_var: float):
    ln_x = ln_evidence_inv_var - 2.0 * ln_evidence_inv
    x = jnp.exp(ln_x)
    ln_evidence = jnp.log(1.0 + x) - ln_evidence_inv
    ln_evidence_std = 0.5 * ln_evidence_inv_var - 2.0 * ln_evidence_inv
    return (ln_evidence, ln_evidence_std)


def ln_bayes_factor(ln_evidence_inv1: float, ln_evidence_inv_var1: float,
                    ln_evidence_inv2: float, ln_evidence_inv_var2: float):

    evidence_inv_ev1 = jnp.exp(ln_evidence_inv1)
    evidence_inv_var_ev1 = jnp.exp(ln_evidence_inv_var1)

    evidence_inv_ev2 = jnp.exp(ln_evidence_inv2)
    evidence_inv_var_ev2 = jnp.exp(ln_evidence_inv_var2)

    common_factor = 1.0 + evidence_inv_var_ev1 / (evidence_inv_ev1**2)

    bf12 = ln_evidence_inv2 - ln_evidence_inv1 + jnp.log(common_factor)

    bf12_std = 0.5*jnp.log(
        evidence_inv_ev1**2 * evidence_inv_var_ev2
        + evidence_inv_ev2**2 * evidence_inv_var_ev1
    ) - 2.0 * ln_evidence_inv1
    return bf12, bf12_std


def estimate_neg_log_evidence(trace,
                              log_probabilities,
                              temperature: float,
                              num_epochs: int,
                              nchains: int = 4
                              ):
    ndim = trace.shape[1]
    chains = hm.Chains(ndim)
    chains.add_chains_2d(trace, log_probabilities, nchains)
    chains_train, chains_infer = hm.utils.split_data(
        chains, training_proportion=0.5
    )
    model = hm.model.RQSplineModel(
        ndim, standardize=True, temperature=temperature
    )
    model.fit(chains_train.samples, epochs=num_epochs, verbose=False)
    ev = hm.Evidence(chains_infer.nchains, model)
    ev.add_chains(chains_infer)
    return ev.ln_evidence_inv, ev.ln_evidence_inv_var


def batch_harmonic(parquet_path: Path, temperature=0.8, num_epochs=20):
    output_path = parquet_path.parent / 'nle' / parquet_path.name
    output_path.parent.mkdir(exist_ok=True)
    if output_path.exists():
        return
    jax.clear_caches()
    df_mcmc = pl.read_parquet(parquet_path)
    df_evidence = []
    for k in range(len(df_mcmc)):
        row = df_mcmc.slice(k, 1)
        sid = row["sourceid"].item()
        posterior = row['log_posterior'].item().to_numpy()
        trace = row.drop(
            ['sourceid', 'log_posterior', r'^n_eff_.*$',  r'^r_hat_.*$']
        ).explode(pl.col('*')).to_numpy()
        ln_inv_ev, ln_inv_ev_var = estimate_neg_log_evidence(
            trace, posterior, temperature=temperature, num_epochs=num_epochs
        )
        df_evidence.append(
            {
                'sourceid': sid,
                'ln_inv_ev': ln_inv_ev,
                'ln_inv_ev_var': ln_inv_ev_var
            }
        )
    pl.DataFrame(df_evidence).write_parquet(output_path, compression_level=22)


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(
        description='AGNs evidence from, posteriors'
    )
    parser.add_argument('path_to_posteriors', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    path_data = Path(args.path_to_posteriors)
    parquet_list = sorted(list((path_data).glob('*.parquet')))
    print(f"Found {len(parquet_list)} files, starting run")

    apply_in_parallel(
        batch_harmonic,
        parquet_list,
        n_jobs=args.n_jobs,
        description='Computing evidences with harmonic'
    )
