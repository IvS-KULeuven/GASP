import jax
from jax import numpy as jnp
from jax import random
import numpy as np
from numpyro.infer import autoguide
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.optim import ClippedAdam

def set_guide(model, guide_type):
    if guide_type == "diag":
        guide = autoguide.AutoDiagonalNormal(model)
    elif guide_type == "full":
        guide = autoguide.AutoMultivariateNormal(model)
    elif guide_type == 'BNAF':
        guide = autoguide.AutoBNAFNormal(model, num_flows=2)
    elif guide_type == 'MAP':
        guide = autoguide.AutoDelta(model)
    elif guide_type == 'Laplace':
        guide = autoguide.AutoLaplaceApproximation(model)
    else:
        raise ValueError("guide_type must be 'diag', 'full' or 'BNAF'")
    return guide

def fit_vi(model, rng_key, x, yerr, y, x_interp=None, 
           guide_type: str = "BNAF", lr: float = 1e-3, max_steps: int = 10_000, num_particles: int = 5,
           early_stopping: bool = False, patience: int = 1000, min_delta: float = 1e-12):
    guide = set_guide(model, guide_type)
    optimizer = ClippedAdam(step_size=lr, clip_norm=5.0)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=num_particles))
    svi_state0 = svi.init(rng_key, x=x, yerr=yerr, y=y, x_interp=x_interp)
    dtype = x.dtype
    
    def body(state, _):
        state, loss = svi.update(state, x=x, yerr=yerr, y=y, x_interp=x_interp)
        return state, loss

    def run_with_early_stopping(svi_state):
        losses = jnp.full((max_steps,), jnp.nan, dtype=dtype)
        step0 = jnp.array(0, dtype=jnp.int32)
        best0 = jnp.array(jnp.inf, dtype=dtype)
        wait0 = jnp.array(0, dtype=jnp.int32)

        def cond(carry):
            state, step, best, wait, losses = carry
            cont = (step < max_steps) & (wait < patience)
            return cont

        def body(carry):
            state, step, best, wait, losses = carry
            state, loss = svi.update(state, x=x, yerr=yerr, y=y, x_interp=x_interp)
            losses = losses.at[step].set(loss.astype(losses.dtype))
            improved = loss < (best - min_delta)
            best = jnp.where(improved, loss, best)
            wait = jnp.where(improved, jnp.array(0, wait.dtype), wait + 1)
            step = step + 1
            return (state, step, best, wait, losses)

        state_f, steps_done, best_f, wait_f, losses_f = jax.lax.while_loop(
            cond, body, (svi_state, step0, best0, wait0, losses)
        )
        return state_f, losses_f, steps_done

    if early_stopping:
        state_f, losses_f, steps_done = jax.jit(run_with_early_stopping)(svi_state0)
        losses_f = losses_f[:steps_done]
    else:
        state_f, losses_f = jax.lax.scan(jax.jit(body), svi_state0, xs=None, length=max_steps)
    params = svi.get_params(state_f)
    return guide, params, losses_f

def prime_laplace_guide(model, vi_params, x, yerr, y):
    guide = set_guide(model, guide_type='Laplace')
    optimizer = ClippedAdam(step_size=1e-3, clip_norm=5.0)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=1))
    svi_state0 = svi.init(random.PRNGKey(0), x=x, yerr=yerr, y=y, x_interp=None)
    return guide
    
def sample_posterior_laplace(model, rng_key, vi_params, x, yerr, y, num_samples: int = 10_000):
    guide = prime_laplace_guide(model, vi_params, x, yerr, y)
    return guide.sample_posterior(rng_key, vi_params, sample_shape=(num_samples,))

def extract_laplace_uncertainty(guide, vi_params):
    transform  = guide.get_transform(vi_params)
    loc        = np.array(transform.loc)
    scale_tril = np.array(transform.scale_tril)
    cov        = scale_tril @ scale_tril.T
    std        = np.sqrt(np.diag(cov))
    return loc, scale_tril, std

def sample_posterior_vi(model, rng_key, x, yerr, y, x_interp=None,
                        guide_type: str = "BNAF", lr: float = 1e-3, max_steps: int = 10_000, num_particles: int = 5,
                        early_stopping: bool = False, num_samples: int = 10_000) -> tuple[dict[str, np.ndarray], np.ndarray]:
    _, rng_key2 = jax.random.split(rng_key, 2)
    guide, vi_params, losses = fit_vi(model, rng_key, x=x, yerr=yerr, y=y, x_interp=x_interp,
                                      guide_type=guide_type, lr=lr, max_steps=max_steps, num_particles=num_particles, early_stopping=early_stopping)
    if guide_type == 'MAP': # MAP are point estimates
        return {k.split('_auto')[0]: np.asarray(v) for k, v in vi_params.items()}, np.asarray(losses)
    predictive = Predictive(guide, params=vi_params, num_samples=num_samples)
    posterior_samples = predictive(rng_key2)
    posterior_samples = {k: np.asarray(v) for k, v in posterior_samples.items() if 'auto' not in k}
    return posterior_samples, np.asarray(losses)


