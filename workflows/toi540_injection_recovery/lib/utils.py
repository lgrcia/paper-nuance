from tinygp import kernels, GaussianProcess
import jax.numpy as jnp
import numpy as np


def poly_X(x, order):
    X = x ** np.arange(0, order)[:, None]
    X[1::] -= X[1::].mean(1)[:, None]
    X[1::] /= X[1::].std(1)[:, None]

    return X


def build_gp(params, x):
    periodic = kernels.quasisep.SHO(
        omega=jnp.exp(params["log_omega"]),
        quality=jnp.exp(params["log_quality"]),
        sigma=jnp.exp(params["log_sigma"]),
    )
    return GaussianProcess(
        periodic, x, diag=jnp.exp(2 * params["log_jitter"]), mean=0.0
    )
