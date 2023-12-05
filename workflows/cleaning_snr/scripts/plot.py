import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binned_statistic_2d

sys.path.append("./lib")
import matplotlib.pyplot as plt
from make_lightcurve import make_lc
from make_params import make
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils import depth as true_depth
from utils import error

df = pd.read_csv(snakemake.input[0])

n = 3  # number of small plots
scale = 1.2
W = n + 1
H = n + 1

snr_lim = 6

fig = plt.figure(None, (7, 6))

def I(i, j):
    return int(j * W + i) + 1


for i in range(H * W):
    ax = plt.subplot(H, W, i + 1)

## Main plot
main = plt.subplot(H, W, (I(0, 1), I(W - 2, H - 1)))

bins = (30, 30)

if df is not None:
    snr, var, amp = df.values.T.astype(float)
    stats = binned_statistic_2d(var, amp, snr, bins=bins)
    im = main.imshow(
        stats.statistic.T,
        origin="lower",
        extent=(
            stats.x_edge.min(),
            stats.x_edge.max(),
            stats.y_edge.min(),
            stats.y_edge.max(),
        ),
        aspect="auto",
        vmax=snr_lim,
        cmap="Greys_r",
    )

main.set_ylabel(
    r"$\delta$ (amplitude)",
    fontsize=14,
)
main.set_xlabel(
    r"$\tau$ (timescale)",
    fontsize=14,
)
params = true_depth
original_snr = true_depth / np.sqrt(np.mean(error) ** 2 / n)

main.set_title(f"Detrending effect on SNR", loc="left")

if df is not None:
    main.text(
        0.1,
        stats.y_edge.max() + 0.26,
        f"original SNR: {original_snr:.2f}",
        color="0.3",
        va="center",
    )

# removing some axes
# ------------------
for i in range(1, W*H):
    ax = plt.subplot(H, W, i)
    plt.axis("off")

# light curves examples
# ---------------------
# zoom box params
a = (0.5, 1)  # xy of lower left box
b = (10, 5)  # xy of upper right box
pcolor = "0.7"
amp = np.linspace(0.5, 3.5, 4)[::-1]
var = np.linspace(2, b[0] - 1, 3)
ymax = 0.06
seed = 10


def title(ax, v=None, a=None):
    title = rf"$\tau = {v:.1f}$" if v is not None else rf"$\delta = {a:.1f}$"
    ax.set_title(title, fontsize=10)


for i, a in enumerate(amp[1::]):
    ax = plt.subplot(H, W, I(W - 1, i + 1))
    v = b[0] - 1
    params = make(delta_v=a, tau_v=v, seed=seed)
    x, y, e = make_lc(params, seed)
    ax.plot(x, y, c="0.5")
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-0.1, np.max(x))
    plt.axis("off")
    title(ax, a=a)


for i, v in enumerate(var):
    ax = plt.subplot(H, W, I(i, 0))
    a = amp[0]
    params = make(delta_v=a, tau_v=v, seed=seed)
    x, y, e = make_lc(params, seed)
    plt.plot(x, y, c="0.5")
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(0.2, np.max(x) * 1.1)
    plt.axis("off")
    title(ax, v=v)

fig.tight_layout(pad=2)
axins = main.inset_axes((0.62, 1.03, 0.3, 0.06))

if df is not None:
    cb = fig.colorbar(im, cax=axins, orientation="horizontal", ticks=[])
    cb.ax.text(-0.4, 0.5, "0", va="center", ha="right")
    cb.ax.text(snr_lim + .4, 0.5, f"> {snr_lim}", va="center", ha="left")
    cb.ax.text(3, 1.5, "SNR", va="center", ha="right")

plt.savefig(snakemake.output[0])
