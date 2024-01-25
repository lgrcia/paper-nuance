from wotan import flatten
import numpy as np
from nuance.utils import transit
import tinygp
import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from utils import snr

seed = 42

# Simulate light curve
# --------------------
np.random.seed(seed)
# transit
time = np.linspace(-1.5, 1.5, 1000)
duration = 0.1
depth = 0.8e-2
n = duration / np.median(np.diff(time))
t0 = 0.0
n = duration / np.median(np.diff(time))

signal = transit(time, time.mean(), duration, depth, c=5000) + 1.0

# white noise
wsigma = 1.5e-3
white_noise = np.random.normal(0, wsigma, len(time))
original_flux = white_noise + signal
original_snr = snr(time, original_flux, t0, duration)

# Simulate variability
# --------------------
lightcurves = []
variabilities = []
gps = []
cond_gps = []

periods = [1000.0, 1.0, 0.2, 0.1, 0.04]
amps = [0.0, 0.8e-2, 0.8e-2, 0.8e-2, 0.8e-2]

for i, (period, amp) in enumerate(zip(periods, amps)):
    Q = 10.0
    w0 = 1 / period
    S0 = amp

    var_kernel = tinygp.kernels.quasisep.SHO(w0, Q, S0)
    gp = tinygp.GaussianProcess(var_kernel, time, mean=0.0)
    cond_gp = tinygp.GaussianProcess(var_kernel, time, mean=0.0, diag=wsigma**2)
    variability = gp.sample(jax.random.PRNGKey(seed))
    variability -= np.median(variability)
    flux = white_noise + signal + variability
    lightcurves.append(flux)
    variabilities.append(variability)
    gps.append(gp)
    cond_gps.append(cond_gp)

# Trends and SNR
# -------------
wotan_trend_snr = []
gp_trend_snr = []

cut = 200
intransit = signal < (1.0 - depth / 2)

for i, (flux, var, gp) in enumerate(zip(lightcurves, variabilities, cond_gps)):
    # WOTAN
    flatten_flux, flatten_trend = flatten(
        time, flux, window_length=3 * duration, return_trend=True
    )
    SNR = snr(time, flatten_flux, t0, duration)
    wotan_trend_snr.append((flatten_trend, SNR))

    # GP
    _, cond = gp.condition(y=flux - 1.0, X_test=time)
    variability_model = cond.mean + 1.0
    flatten_flux = flux - variability_model + 1.0
    SNR = snr(time, flatten_flux, t0, duration)
    gp_trend_snr.append((variability_model, SNR))


# Plots
# -----
offset = 0.05

wotan_color = "C4"
gp_color = "C3"

intransit = signal < (1.0 - depth / 2)

x_wotan = 3
x_gp = 6.2

snr_offset = 0.6

from matplotlib.pyplot import GridSpec

fig = plt.figure(figsize=(8.5, 6))
gs = GridSpec(5, 5, figure=fig, width_ratios=[1, 0.6, 0.2, 0.6, 0.2], hspace=0)

ylim = (0.96, 1.03)

for i, (flux, var, gp) in enumerate(zip(lightcurves, variabilities, gps)):
    ax = plt.subplot(gs[i, 0])
    ax_wotan = plt.subplot(gs[i, 1])
    ax_gp = plt.subplot(gs[i, 3])
    ax.plot(time, flux, ".", c="0.8")
    if i == 0:
        ax.plot(time, signal, c="k", alpha=0.5, lw=2)

    # wotan
    trend, _snr = wotan_trend_snr[i]
    if i != 0:
        ax.plot(time, trend, c=wotan_color, alpha=0.6, lw=2)
        plt.subplot(gs[i, 2]).text(0.5, 0.5, f"{_snr:.2f}", ha="center", fontsize=12)
        ax_wotan.plot(
            time,
            (flux - trend) + 1.0,
            ".",
            c="0.8",
        ),

    # gp
    trend, _snr = gp_trend_snr[i]
    if i != 0:
        ax.plot(time, trend, c=gp_color, alpha=0.6, lw=2)
        plt.subplot(gs[i, 4]).text(0.5, 0.5, f"{_snr:.2f}", ha="center", fontsize=12)
        ax_gp.plot(
            time,
            (flux - trend) + 1.0,
            ".",
            c="0.8",
        )

    if i != 0:
        ax.plot(time, var + 1, c="C0", alpha=0.6, lw=2)

    ax.set_ylim(ylim)
    ax_wotan.set_ylim(ylim)
    ax_gp.set_ylim(ylim)

    xlim = np.array([-1, 1]) * 0.6 + t0
    ax_wotan.set_xlim(xlim)
    ax_gp.set_xlim(xlim)

    if i != 0:
        ax.text(
            -1.5,
            1.03,
            f"$\omega = ${2*np.pi/periods[i]:.0f}",
            ha="left",
            fontsize=11,
            color="0.5",
        )

for i in range(0, 4):
    for j in [0, 1, 3]:
        ax = plt.subplot(gs[i, j])
        ax.axis("off")

for i in range(0, 5):
    for j in [2, 4]:
        ax = plt.subplot(gs[i, j])
        ax.axis("off")

for i in [0, 1, 3]:
    ax = plt.subplot(gs[-1, i])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.subplot(gs[0, 0]).set_title("original light curve", fontsize=12)
plt.subplot(gs[1, 1]).set_title("cleaned with bi-weight", color="C4", fontsize=12)
plt.subplot(gs[1, 3]).set_title("cleaned with GP", color="C3", fontsize=12)
plt.subplot(gs[1, 2]).set_title("SNR", fontsize=12)
plt.subplot(gs[1, 4]).set_title("SNR", fontsize=12)

plt.subplot(gs[-1, 0]).set_xlabel("time (days)")
plt.subplot(gs[-1, 1]).set_xlabel("time (days)")
plt.subplot(gs[-1, 3]).set_xlabel("time (days)")

top_ax = plt.subplot(gs[0:1, 1:5])
top_ax.axis("off")
top_ax.text(
    0.5,
    1,
    f"original SNR\n{original_snr:.2f}",
    ha="center",
    va="top",
    fontsize=12,
    linespacing=1.5,
)

plt.tight_layout()
plt.savefig(snakemake.output[0])