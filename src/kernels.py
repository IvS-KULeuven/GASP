from jax import jit
import jax.numpy as jnp


@jit
def cosine_kernel(x, z, frequency):
    dx = x[:, None] - z
    return jnp.cos(2.0 * jnp.pi * dx * frequency)


@jit
def periodic_kernel(x, z, frequency, lengthscale):
    angle = jnp.pi * jnp.abs(x[:, None] - z) * frequency
    return jnp.exp(-2.*jnp.power(jnp.sin(angle) / lengthscale, 2))


@jit
def linear_kernel(x, z, offset):
    return (x[:, None] - offset)*(z - offset)


@jit
def powered_exponential_kernel(x, z, lengthscale, gamma):
    base = jnp.abs(x[:, None] - z) / lengthscale
    return jnp.exp(-1.0 * jnp.pow(base, gamma))


def cosine_covariance(x, z, hyperparameters):
    variance = hyperparameters['cos_stdev'] ** 2
    frequency = hyperparameters['cos_frequency']
    return variance * cosine_kernel(x, z, frequency)


def periodic_covariance(x, z, hyperparameters):
    variance = hyperparameters['cos_stdev'] ** 2
    frequency = hyperparameters['cos_frequency']
    lengthscale = hyperparameters['cos_length']
    return variance * periodic_kernel(x, z, frequency, lengthscale)


def powered_exponential_covariance(x, z, hyperparameters, gamma=1.0):
    variance = hyperparameters['red_stdev'] ** 2
    lengthscale = hyperparameters['red_length']
    return variance * powered_exponential_kernel(x, z, lengthscale, gamma)


def linear_covariance(x, z, hyperparameters):
    variance = hyperparameters['red_stdev'] ** 2
    offset = hyperparameters['mean']
    return variance * linear_kernel(x, z, offset)


def cosine_plus_exp_covariance(x, z, hyperparameters):
    C1 = cosine_covariance(x, z, hyperparameters)
    C2 = powered_exponential_covariance(x, z, hyperparameters)
    return C1 + C2


def cosine_plus_exp_ratio_covariance(x, z, hyperparameters):
    variance = hyperparameters['cos_stdev'] ** 2
    beta = hyperparameters['beta']
    K1 = powered_exponential_kernel(x, z, hyperparameters['red_length'], gamma=1.0)
    K2 = cosine_kernel(x, z, hyperparameters['cos_frequency'])
    return variance * (beta * K1 + (1.0-beta) * K2)
    return variance * (K1 + K2)


def cosine_plus_linear_covariance(x, z, hyperparameters):
    C1 = cosine_covariance(x, z, hyperparameters)
    C2 = linear_covariance(x, z, hyperparameters)
    return C1 + C2


def cosine_times_exp_covariance(x, z, hyperparameters):
    variance = hyperparameters['cos_stdev'] ** 2
    K1 = powered_exponential_kernel(x, z, hyperparameters['red_length'])
    K2 = cosine_kernel(x, z, hyperparameters['cos_frequency'])
    # K2 = periodic_kernel(x, z, hyperparameters)
    return variance * K1 * K2
