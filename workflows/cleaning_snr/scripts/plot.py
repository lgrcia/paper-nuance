# make_lightcurve.py
import jax
import numpy as np
import tinygp
from nuance.utils import transit
from make_params import make
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def show_df(df, ax):
    bins = (30, 30)

    if df is not None:
        snr, var, amp = df.values.T.astype(float)
        snr = np.maximum(0, snr)
        stats = binned_statistic_2d(var, amp, snr, bins=bins)
        im = ax.imshow(
            stats.statistic.T,
            origin="lower",
            extent=(
                stats.x_edge.min(),
                stats.x_edge.max(),
                stats.y_edge.min(),
                stats.y_edge.max(),
            ),
            aspect="auto",
            vmax=10,
            vmin=5,
            cmap="Greys_r",
        )
        return im
    
config = snakemake.config
true_depth = config["depth"]
error = config["error"]
time = np.arange(0, config["length"], config["exposure"])


jax.config.update("jax_enable_x64", True)


def build_gp(time, params):
    kernel = tinygp.kernels.quasisep.SHO(
        params["omega"], params["quality"], sigma=params["sigma"]
    )
    gp = tinygp.GaussianProcess(kernel, time)
    return gp


def make_lc(time, params, seed):
    gp = build_gp(time, params)

    signal = transit(
        time, time.mean(), params["duration"], params["depth"], c=50000, P=None
    )
    y = gp.sample(jax.random.PRNGKey(seed)) + signal

    error = np.ones_like(y) * params["error"]

    return y, y + np.random.randn(len(y)) * error


deltas = np.linspace(0.5, 4, 3)[::-1]
taus = np.linspace(2, 5, 4)
time = np.arange(0, config["length"], config["exposure"])
t0 = time.mean()
transit_signal = (np.abs(t0 - time) < config["duration"] / 2).astype(float) * -1

taus_signals = []
for t in taus:
    params = make(config, delta_v=1, tau_v=t, seed=10)
    taus_signals.append(make_lc(time, params, 10))

deltas_signals = []
for d in deltas:
    params = make(config, delta_v=d, tau_v=5, seed=10)
    deltas_signals.append(make_lc(time, params, 10))

fig = plt.figure(None, (8, 4))
gs = GridSpec(
    4,
    5,
)

for i, signal in enumerate(taus_signals):
    ax = plt.subplot(gs[0, i])
    ax.axis("off")
    ax.plot(time, signal[1], ".", c="0.8", ms=1.5)
    ax.plot(time, signal[0], c="0.4", lw=1)
    ax.set_ylim(-0.01, 0.01)
    ax.set_xlim(t0 - 0.8, t0 + 0.8)
    ax.set_title(r"$\tau = {}$".format(taus[i]), fontsize=10)

for i, signal in enumerate(deltas_signals):
    ax = plt.subplot(gs[1 + i, -1])
    ax.axis("off")
    ax.plot(time, signal[1], ".", c="0.8", ms=1.5)
    ax.plot(time, signal[0], c="0.4", lw=1)
    ax.set_ylim(-0.01, 0.01)
    ax.set_xlim(t0 - 0.8, t0 + 0.8)
    ax.set_title(r"$\delta = {}$".format(deltas[i]), fontsize=10)

bi_ax = plt.subplot(gs[1::, 0:2])
df = pd.read_csv(snakemake.input[0])
_ = show_df(df, bi_ax)
bi_ax.text(0.05, 0.9, "bi-weight", transform=bi_ax.transAxes, color="w", fontsize=11)
bi_ax.set_ylabel(r"$\delta$")
bi_ax.set_xlabel(r"$\tau$")

gp_ax = plt.subplot(gs[1::, 2:4])
df = pd.read_csv(snakemake.input[1])
im = show_df(df, gp_ax)
gp_ax.text(0.05, 0.9, "GP", transform=gp_ax.transAxes, color="w", fontsize=11)
gp_ax.set_yticklabels([])
gp_ax.set_xlabel(r"$\tau$")
axins = gp_ax.inset_axes((0.62, 1.03, 0.3, 0.06))
cb = fig.colorbar(im, cax=axins, orientation="horizontal", ticks=[])
cb.ax.text(4.7, 0.5, "SNR: 5", va="center", ha="right")
cb.ax.text(10.4, 0.5, "10", va="center", ha="left")

plt.tight_layout()

plt.savefig(snakemake.output[0])
