import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib import patches

nuance_main, wotan_main, nuance_zoom, wotan_zoom = snakemake.input

ylabel = r"$\delta$"
xlabel = r"$\tau$"

fig = plt.figure(None, (7, 6.5))

bins = (30,30)

# wtls
# ----
main = plt.subplot(2, 2, 1)
main.set_ylabel(ylabel, fontsize=14)
df = pd.read_csv(wotan_main)
detected, _var, _amp = df.values.T.astype(float)
stats = binned_statistic_2d(_var, _amp, detected, bins=bins)
im = main.imshow(
    stats.statistic.T, 
    origin="lower",
    extent=(
        stats.x_edge.min(), 
        stats.x_edge.max(), 
        stats.y_edge.min(), 
        stats.y_edge.max()
    ),
    aspect="auto",
    vmin=0.,
    vmax=1.
)
main.set_title("wõtan + TLS\n\n")


zoomw = plt.subplot(2, 2, 3)
bins = (30,30)
df = pd.read_csv(wotan_zoom)
detected, _var, _amp = df.values.T.astype(float)
stats = binned_statistic_2d(_var, _amp, detected, bins=bins)
zoomw.imshow(
    stats.statistic.T, 
    origin="lower",
    extent=(
        stats.x_edge.min(), 
        stats.x_edge.max(), 
        stats.y_edge.min(), 
        stats.y_edge.max()
    ),
    aspect="auto",
    vmin=0.,
    vmax=1.
)

a = (0.01,0.01) # xy of lower left box
b = (10, 25) # xy of upper right box
pcolor="0.6"
p = patches.Polygon([a, (a[0], b[1]), (b[0], b[1]), (b[0], a[1])], fill=False, ec=pcolor)
main.add_patch(p)
con = patches.ConnectionPatch(xyA=(a[0], b[1]), xyB=(0.1, b[1]-0.1), coordsA="data", coordsB="data", axesA=main, axesB=zoomw, color=pcolor)
fig.add_artist(con)
con = patches.ConnectionPatch(xyA=(b[0], a[1]), xyB=(b[0], b[1]-0.1), coordsA="data", coordsB="data", axesA=main, axesB=zoomw, color=pcolor)
fig.add_artist(con)
zoomw.set_ylim(0.1, b[1])
zoomw.set_xlim(0.1, b[0])
zoomw.set_xlabel(xlabel, fontsize=14)
zoomw.set_ylabel(ylabel, fontsize=14)

# nuance
# ------
main = plt.subplot(2, 2, 2)
bins = (30,30)
df = pd.read_csv(nuance_main)
detected, _var, _amp = df.values.T.astype(float)
stats = binned_statistic_2d(_var, _amp, detected, bins=bins)
main.imshow(
    stats.statistic.T, 
    origin="lower",
    extent=(
        stats.x_edge.min(), 
        stats.x_edge.max(), 
        stats.y_edge.min(), 
        stats.y_edge.max()
    ),
    aspect="auto",
    vmin=0.,
    vmax=1.
)
main.set_title("nuance\n")
axins = main.inset_axes((0.455, 1.03, 0.4, 0.06))

zoom = plt.subplot(2, 2, 4)
df = pd.read_csv(nuance_zoom)
detected, _var, _amp = df.values.T.astype(float)
stats = binned_statistic_2d(_var, _amp, detected, bins=bins)
zoom.imshow(
    stats.statistic.T, 
    origin="lower",
    extent=(
        stats.x_edge.min(), 
        stats.x_edge.max(), 
        stats.y_edge.min(), 
        stats.y_edge.max()
    ),
    aspect="auto",
    vmin=0.,
    vmax=1.
)

pcolor="0.6"
p = patches.Polygon([a, (a[0], b[1]), (b[0], b[1]), (b[0], a[1])], fill=False, ec=pcolor)
main.add_patch(p)
con = patches.ConnectionPatch(xyA=(a[0], b[1]), xyB=(0.1, b[1]-0.1), coordsA="data", coordsB="data", axesA=main, axesB=zoom, color=pcolor)
fig.add_artist(con)
con = patches.ConnectionPatch(xyA=(b[0], a[1]), xyB=(b[0], b[1]-0.1), coordsA="data", coordsB="data", axesA=main, axesB=zoom, color=pcolor)
fig.add_artist(con)
zoom.set_ylim(0.1, b[1])
zoom.set_xlim(0.1, b[0])
zoom.set_xlabel(xlabel, fontsize=14)

plt.tight_layout()

cb = fig.colorbar(im, cax=axins, orientation="horizontal", ticks=[])
cb.ax.text(-0.05, 0.5, "0%", va="center", ha="right")
cb.ax.text(1.06, 0.5, "100%", va="center", ha="left")
cb.ax.text(-0.3, 0.5, "recovery", va="center", ha="right")
plt.savefig(snakemake.output[0])