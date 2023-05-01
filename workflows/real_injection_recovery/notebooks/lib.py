from tinygp import kernels
import jax
import jaxopt
import numpy as np
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def plot_kernel(kernel, lag=None, **kwargs):
    if lag is None:
        lag = np.linspace(0, 30, 1000)
    k = kernel(lag, lag[:1])
    plt.plot(lag, k/k[0], **kwargs)
    plt.xlabel("dx")
    plt.ylabel("k(dx)")
    
def print_params(params):
    return {n.split("log_")[-1] if "log" in n else n: f"{np.exp(p):.2e}" if "log" in n else p for n, p in params.items()}

# GP
def build_gp(params, x):
    
    short = kernels.quasisep.Matern32(sigma=jnp.exp(params["log_sigma_short"]), scale=jnp.exp(params["log_scale_short"]))
    long = kernels.quasisep.Matern32(sigma=jnp.exp(params["log_sigma_long"]), scale=jnp.exp(params["log_scale_long"]))
    periodic = kernels.quasisep.SHO(omega=jnp.exp(params["log_omega"]), quality=jnp.exp(params["log_quality"]), sigma=jnp.exp(params["log_sigma"]))
    
    kernel = periodic + short + long
    
    return GaussianProcess(kernel, x, diag=jnp.exp(2 * params["log_jitter"]), mean=0.)

@jax.jit
def solve(x, y, X, solver):
    Liy = solver.solve_triangular(y)
    LiX = solver.solve_triangular(X.T)
    LiXT = LiX.T
    w = jax.numpy.linalg.lstsq(LiXT@LiX, LiXT@Liy)[0]
    v = jax.numpy.linalg.lstsq(LiXT@LiX, jnp.ones_like(w))[0]
    return w, v

@jax.jit
def neg_log_likelihood(params, x, y, X):
    gp = build_gp(params, x)
    w = solve(x, y, X, gp.solver)[0]
    return - gp.log_probability(y - w@X)

@jax.jit
def _eval(params, x, y, X):
    gp = build_gp(params, x)
    w, v = solve(x, y, X, gp.solver)
    return - gp.log_probability(y - w@X), w, v
    
def model(x, y, X, mask=None):
    if mask is None:
        mask = mask = np.ones_like(x).astype(bool)
        
    masked_x = x[mask]
    masked_y = y[mask]
    masked_X = X[:, mask]
    
    @jax.jit
    def mu(params):
        gp = build_gp(params, masked_x)
        w = solve(masked_x, masked_y, masked_X, gp.solver)[0]
        cond_gp = gp.condition(masked_y - w@masked_X, x).gp
        return cond_gp.loc + w@X
    
    def optimize(init_params, param_names=None):
        def inner(theta, *args, **kwargs):
            params = dict(init_params, **theta)
            return neg_log_likelihood(params, *args, **kwargs)

        param_names = list(init_params.keys()) if param_names is None else param_names
        start = {k: init_params[k] for k in param_names}

        solver = jaxopt.ScipyMinimize(fun=inner)
        soln = solver.run(start, x=masked_x, y=masked_y, X=masked_X)
        print(soln.state)

        return dict(init_params, **soln.params)
    
    def plot_kernel(params, lag=None, **kwargs):
        gp = build_gp(params, masked_x)
        if lag is None:
            lag = np.linspace(0, 30, 1000)
        k = gp.kernel(lag, lag[:1])
        plt.plot(lag, k/k[0], **kwargs)
        plt.xlabel("dx")
        plt.ylabel("k(dx)")
    
    return optimize, mu, plot_kernel


def iterative_opt(params, model, x, y, X, it=3, upper_sigma=4, lower_sigma=3, n_up=15):
    
    mask = np.ones_like(x).astype(bool)

    new_params = params.copy()

    for i in range(it):
        if i == 0:
            m = np.mean(flux)
        else:
            m = mu(new_params).__array__()
        r = (flux - m)
        mask_up = r < np.std(r[mask])*upper_sigma
        mask_down = r > - np.std(r[mask])*lower_sigma

        # mask around flares
        ups = np.flatnonzero(~mask_up)
        if len(ups) > 0:
            mask_up[np.hstack([np.arange(max(u-n_up, 0), min(u+n_up, len(time))) for u in ups])] = False
        mask = np.logical_and(mask_up, mask_down)

        optimize, mu, plot_kernel = model(x, y, X, mask=mask)
        new_params = optimize(new_params)
    
    return new_params