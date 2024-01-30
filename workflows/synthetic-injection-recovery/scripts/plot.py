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

methods = ["bspline", "biweight", "harmonics", "iterative", "gp", "nuance"]
for method in methods:
    df[method] = df.apply(lambda row: is_found(row[method], row["period"]), axis=1)


bins = (22, 22)
fig = plt.figure(None, (8, 5.))
cmap = "Greys_r"

titles = {
    "bspline": "Bspline + BLS",
    "biweight": "bi-weight + BLS",
    "harmonics": "harmonics + BLS",
    "iterative": "iterative + BLS",
    "gp": "GP + BLS",
    "nuance": "nuance",
}

for i, method in enumerate(methods):
    ax = plt.subplot(2, 3, i + 1)
    tau, delta, found = df[["tau", "delta", method]].values.T
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
        vmin=0,
        vmax=1,
    )
    ax.set_title(titles[method])

for i in [2, 3, 5, 6]:
    ax = plt.subplot(2, 3, i)
    ax.set_yticklabels([])

for i in [1, 2, 3]:
    ax = plt.subplot(2, 3, i)
    ax.set_xticklabels([])
    ax = plt.subplot(2, 3, i + 3)
    ax.set_xlabel(r"$\tau$")

for i in [1, 4]:
    ax = plt.subplot(2, 3, i)
    ax.set_ylabel(r"$\delta$")


ax = plt.subplot(2, 3, 3)
ax.text(
    0.05,
    0.92,
    f"{len(tau)} samples (binned)",
    va="center",
    ha="left",
    transform=ax.transAxes,
    color="w",
)

ax = plt.subplot(2, 3, 6)
axins = ax.inset_axes((1.07, 0.05, 0.05, 0.4))
ax.set_yticklabels([])
cb = fig.colorbar(im, cax=axins, orientation="vertical", ticks=[])
cb.ax.text(-0.1, -0.2, "0%", va="center", ha="left")
cb.ax.text(-0.1, 1.2, "100%", va="center", ha="left")
cb.ax.text(1.3, 0.5, "recovery", va="center", ha="left", rotation=-90)

plt.tight_layout()
plt.tight_layout()
plt.savefig(snakemake.output[0])
