import numpy as np
import matplotlib.pyplot as plt
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import tinygp
from utils import snr

# Simulation
# ------------
from nuance.utils import transit

np.random.seed(43)

fluxes = []
snrs = []

# transit
duration = 1 / 24
depth = 0.5e-2
total_time = 1
time = np.arange(0, total_time, 1 / 60 / 24)
t0 = time.mean()
n = duration / np.median(np.diff(time))
signal = transit(time, t0, duration, depth, c=5000) + 1.0

# white noise
w = 1.5e-3
white_noise = np.random.normal(0, w, len(time))
flux = white_noise + signal

# No systematics
# --------------

SNR = snr(time, flux, t0, duration)

fluxes.append(flux.copy())
snrs.append(SNR.copy())

# Systematics
# -----------

kernel = tinygp.kernels.quasisep.SHO(60.0, 0.5, depth / 4)
gp = tinygp.GaussianProcess(kernel, time, diag=0.0, mean=0.0)
red_noise = gp.sample(jax.random.PRNGKey(42))
flux = white_noise + red_noise + signal

SNR = snr(time, flux, t0, duration)

fluxes.append(flux.copy())
snrs.append(SNR.copy())

# Plot
# ----

plt.figure(figsize=(6, 5))
offset = 0.02
xoff = time.max() * 1.2

plt.text(xoff, 1.008, f"transit SNR", ha="center", fontsize=14, va="center")
plt.text(time.mean(), 1.008, f"Light curve", ha="center", fontsize=14)

plt.plot(time, fluxes[0], ".", c="0.8")
plt.plot(time, signal, c="k", alpha=0.4, lw=2)
plt.text(
    time.mean(), 1 - 2.2 * depth, "white noise + transit", ha="center", fontsize=14
)
plt.text(xoff, 1, f"{snrs[0]:.2f}", ha="center", fontsize=14, va="center")

plt.plot(time, fluxes[1] - offset, ".", c="0.8")
plt.plot(time, red_noise + 1 - offset, c="C3", alpha=0.5, lw=2)
plt.text(
    time.mean(),
    1 - offset - 2.2 * depth,
    "+ correlated noise",
    ha="center",
    fontsize=14,
    c="C3",
)
plt.text(xoff, 1 - offset, f"{snrs[1]:.2f}", ha="center", fontsize=14, va="center")

plt.axis("off")
plt.tight_layout()
plt.savefig("figures/issue1.pdf")