import numpy as np
import matplotlib.pyplot as plt
import jax, jaxopt
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
import matplotlib.pyplot as plt


def white_red(time, flux, error, plot=False, mean=0.):
    init = {
        "log_diag": 2*np.log(np.mean(error)),
        "log_scale": np.log(1), # 5 mins
        "log_amp": np.log(np.std(flux))
    }

    def build_gp(params):
        kernel = jnp.exp(params["log_amp"]) * kernels.ExpSquared(jnp.exp(params["log_scale"]))
        return GaussianProcess(kernel, time, diag=jnp.exp(params["log_diag"]), mean=mean)

    @jax.jit
    def loss(params):
        gp = build_gp(params)
        return -gp.log_probability(flux)
    
    opt = jaxopt.ScipyMinimize(fun=loss)
    params = opt.run(init).params  
    w = np.sqrt(np.exp(params['log_diag']))
    
    if plot:
        gp = build_gp(params)
        _, cond = gp.condition(y=flux, X_test=time)
        mu = cond.mean
        plt.plot(time, flux, ".", c="0.8")
        plt.plot(time, mu, c="k")
        noise = w
        plt.fill_between(time, mu + noise, mu - noise, color="C0", alpha=0.3)

    return w, np.sqrt(np.exp(params['log_amp']))