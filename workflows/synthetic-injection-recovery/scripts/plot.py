import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d


def is_found(period, true_period):
    period_check = np.abs(period - true_period) < 0.01
    alias_check = np.abs(2 * period - true_period) < 0.01
    alias_check |= np.abs(period / 2 - true_period) < 0.01
    if period_check:
        return 1
    elif alias_check:
        return 1
    else:
        return 0


df = pd.read_csv(snakemake.input[0])

df["bls_found"] = df.apply(
    lambda row: is_found(row["bls_period"], row["period"]), axis=1
)
df["nuance_found"] = df.apply(
    lambda row: is_found(row["nuance_period"], row["period"]), axis=1
)

bins = (22, 22)
fig = plt.figure(None, (7.5, 3.5))
cmap = "Greys_r"

ax = plt.subplot(121)
bins = (22, 22)
tau, delta, found = df[["tau", "delta", "bls_found"]].values.T
stats = binned_statistic_2d(tau, delta, found, bins=bins)
im = plt.imshow(
    stats.statistic.T,
    origin="lower",
    extent=(
        stats.x_edge.min(),
        stats.x_edge.max(),
        stats.y_edge.min(),
        stats.y_edge.max(),
    ),
    aspect="auto",
    cmap=cmap,
)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$\delta$")
ax.set_title("biweight+BLS")

ax.text(
    0.05,
    0.92,
    f"{len(tau)} samples (binned)",
    va="center",
    ha="left",
    transform=ax.transAxes,
    color="w",
)

ax = plt.subplot(122)
bins = (22, 22)
tau, delta, found = df[["tau", "delta", "nuance_found"]].values.T
stats = binned_statistic_2d(tau, delta, found, bins=bins)
im = plt.imshow(
    stats.statistic.T,
    origin="lower",
    extent=(
        stats.x_edge.min(),
        stats.x_edge.max(),
        stats.y_edge.min(),
        stats.y_edge.max(),
    ),
    aspect="auto",
    cmap=cmap,
)
ax.set_xlabel(r"$\tau$")
ax.set_title("nuance")
axins = ax.inset_axes((1.07, 0.05, 0.05, 0.4))
cb = fig.colorbar(im, cax=axins, orientation="vertical", ticks=[])
cb.ax.text(-0.1, -0.12, "0%", va="center", ha="left")
cb.ax.text(-0.1, 1.1, "100%", va="center", ha="left")
cb.ax.text(1.3, 0.5, "recovery", va="center", ha="left", rotation=-90)

plt.tight_layout()
plt.savefig(snakemake.output[0])
