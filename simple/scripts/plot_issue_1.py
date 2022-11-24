import numpy as np
import matplotlib.pyplot as plt
import jax, jaxopt
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tinygp import kernels, GaussianProcess
import matplotlib.pyplot as plt
import tinygp

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

# Simulation
# ------------
from nuance.utils import transit
np.random.seed(43)

fluxes = []
snrs = []

# transit
time = np.linspace(0, 3, 1000)
dur = 0.1
depth = 0.8e-2
n = dur/np.median(np.diff(time))
signal = transit(time, time.mean(), dur, depth, c=10) + 1.

# white noise
wsigma = 1.5e-3
white_noise = np.random.normal(0, wsigma, len(time))
flux = white_noise + signal

# No systematics
# --------------

w, r = white_red(time, flux-signal, wsigma, plot=True)
SNR = depth/np.sqrt((w**2)/n + r**2)

fluxes.append(flux.copy())
snrs.append(SNR.copy())

# Systematics
# -----------

kernel = tinygp.kernels.quasisep.Matern32(0.1, 1.5e-3)
gp = tinygp.GaussianProcess(kernel, time, diag=0., mean=0.)
red_noise = gp.sample(jax.random.PRNGKey(42))
flux =  white_noise + red_noise + signal

w, r = white_red(time, flux-signal, wsigma)
SNR = depth/np.sqrt((w**2)/n + r**2)

fluxes.append(flux.copy())
snrs.append(SNR.copy())

# Variability
# -----------

var_kernel = tinygp.kernels.quasisep.SHO(10, 3, 0.4e-2)
gp = tinygp.GaussianProcess(var_kernel, time, diag=0., mean=0.)
variability = gp.sample(jax.random.PRNGKey(43))

flux = white_noise + red_noise + signal + variability

w, r = white_red(time, flux-signal, wsigma)
SNR = depth/np.sqrt((w**2)/n + r**2)

fluxes.append(flux.copy())
snrs.append(SNR.copy())


# Plot
# ----

plt.figure(figsize=(6,5))
offset = 0.03

plt.text(3.4, 1.01, f"transit SNR", ha="center", fontsize=14, alpha=0.7, va="center")
plt.text(time.mean(), 1.01, f"Light curve", ha="center", fontsize=14, alpha=0.7)

plt.plot(time, fluxes[0], ".", c="0.8")
plt.plot(time, signal, c="k", alpha=0.4, lw=2)
plt.text(time.mean(), 0.985, "white noise + transit", ha="center", fontsize=14)
plt.text(3.4, 1, f"{snrs[0]:.2f}", ha="center", fontsize=12)

plt.plot(time, fluxes[1] - offset, ".", c="0.8")
plt.plot(time, red_noise + 1 - offset, c="C3", alpha=0.5, lw=2)
plt.text(time.mean(), 0.987 - offset, "+ correlated noise", ha="center", fontsize=14, c="C3")
plt.text(3.4, 1 - offset, f"{snrs[1]:.2f}", ha="center", fontsize=12)

plt.axis("off")
plt.tight_layout()
plt.savefig("figures/first_issue.pdf")