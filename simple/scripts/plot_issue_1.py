import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import tinygp

# Simulation
# ------------
from nuance.utils import transit
np.random.seed(43)

fluxes = []
snrs = []

# transit
dur = 1/24
depth = 1e-2
total_time = 1
time = np.arange(0, total_time, 1/60/24)
n = dur/np.median(np.diff(time))
signal = transit(time, time.mean(), dur, depth) + 1.

# white noise
w = 1.5e-3
white_noise = np.random.normal(0, w, len(time))
flux = white_noise + signal
intransit = signal < (1. - depth/2)

# No systematics
# --------------

r = np.max([0, np.std(flux[~intransit]) - w])
SNR = depth/np.sqrt((w**2)/n + r**2)

fluxes.append(flux.copy())
snrs.append(SNR.copy())

# Systematics
# -----------

kernel = tinygp.kernels.quasisep.Matern32(dur, depth/5)
gp = tinygp.GaussianProcess(kernel, time, diag=0., mean=0.)
red_noise = gp.sample(jax.random.PRNGKey(42))
flux =  white_noise + red_noise + signal

r = np.max([0, np.std(flux[~intransit]) - w])
SNR = depth/np.sqrt((w**2)/n + r**2)

fluxes.append(flux.copy())
snrs.append(SNR.copy())

# Plot
# ----

plt.figure(figsize=(6,5))
offset = 0.03
xoff = time.max()*1.2

plt.text(xoff, 1.01, f"transit SNR", ha="center", fontsize=12, va="center")
plt.text(time.mean(), 1.01, f"Light curve", ha="center", fontsize=14, alpha=0.7)

plt.plot(time, fluxes[0], ".", c="0.8")
plt.plot(time, signal, c="k", alpha=0.4, lw=2)
plt.text(time.mean(), 1-1.8*depth, "white noise + transit", ha="center", fontsize=14)
plt.text(xoff, 1, f"{snrs[0]:.2f}", ha="center", fontsize=12, va="center")

plt.plot(time, fluxes[1] - offset, ".", c="0.8")
plt.plot(time, red_noise + 1 - offset, c="C3", alpha=0.5, lw=2)
plt.text(time.mean(), 1 - offset -1.8*depth, "+ correlated noise", ha="center", fontsize=14, c="C3")
plt.text(xoff, 1 - offset, f"{snrs[1]:.2f}", ha="center", fontsize=12, va="center")

plt.axis("off")
plt.tight_layout()
plt.savefig("figures/issue1.pdf")