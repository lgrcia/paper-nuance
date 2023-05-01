import pickle

import pandas as pd
import yaml

df = pd.read_csv(snakemake.input[0])
gp_params = yaml.full_load(open(snakemake.input[1], "r"))
data = pickle.load(open(snakemake.input[2], "rb"))
tic = snakemake.wildcards.target


tls, nuance, radii, periods, tau, delta = (
    df["tls_found"].values,
    df["nuance_found"].values,
    df["planet_radius"].values,
    df["planet_period"].values,
    df["tau"].values,
    df["delta"].values,
)

import numpy as np
from nuance.star import Star

jitter = np.exp(gp_params["log_jitter"])
star = Star(
    data["star_radius"], data["star_mass"], data["star_amplitude"], data["star_period"]
)

unique_periods = np.unique(periods)
min_radii = star.min_radius(unique_periods, 6, len(data["time"]), jitter)


import matplotlib.pyplot as plt
import numpy as np

# setting up ranges
taus_range = np.min(tau), np.max(tau)
deltas_range = np.min(delta), np.max(delta)
periods_range = np.min(periods), np.max(periods)
radii_range = np.min(radii), np.max(radii)
extent = (*periods_range, *radii_range)

# figure
fig = plt.figure(None, (8.5, 3.5))
sec_color = "0.6"

ax = plt.subplot(122)
plt.imshow(
    nuance.reshape(20, 20), extent=extent, origin="lower", aspect="auto", cmap="Greys_r"
)
plt.plot(unique_periods, min_radii, label="SNR = 6")
plt.xlim(periods_range)
plt.ylim(radii_range)
plt.xlabel("Period (days)")
ax.set_title("nuance")
# secondary axes
secax = ax.secondary_yaxis("right")
radii_ticks = ax.get_yticks()
delta_ticks = star.radius2delta(radii_ticks)
secax.set_yticks(radii_ticks, [f"{t:.1f}" for t in delta_ticks])
secax.set_ylabel(r"$\delta$", color=sec_color)
secax.tick_params(axis="y", colors=sec_color)

secax = ax.secondary_xaxis("top")
period_ticks = ax.get_xticks()
tau_ticks = star.period2tau(period_ticks)
secax.set_xticks(period_ticks, [f"{t:.1f}" for t in tau_ticks])
secax.set_xlabel(r"$\tau$", color=sec_color)
secax.tick_params(axis="x", colors=sec_color)
l = ax.legend(loc="upper right", frameon=False)
l.get_texts()[0].set_color("C0")

ax = plt.subplot(121)
plt.imshow(
    tls.reshape(20, 20), extent=extent, origin="lower", aspect="auto", cmap="Greys_r"
)
plt.plot(unique_periods, min_radii, label="SNR = 6")
plt.xlim(periods_range)
plt.ylim(radii_range)
plt.xlabel("Period (days)")
ax.set_title("wõtan + TLS")
plt.ylabel("Radius (R$_\oplus$)")
secax = ax.secondary_xaxis("top")
period_ticks = ax.get_xticks()
tau_ticks = star.period2tau(period_ticks)
secax.set_xticks(period_ticks, [f"{t:.1f}" for t in tau_ticks])
secax.tick_params(axis="x", colors=sec_color)
secax.set_xlabel(r"$\tau$", color=sec_color)

plt.tight_layout()

plt.savefig(snakemake.output[0])
