import pickle

import numpy as np
import yaml
from nuance.star import Star

info = yaml.safe_load(open(snakemake.input.info, "r"))
data = pickle.load(open(snakemake.input.fluxes, "rb"))
n = snakemake.params["n"]

star = Star(
    info["star_radius"], info["star_mass"], info["star_amplitude"], info["star_period"]
)

min_period = 0.4
max_period = 5.0
max_snr = 30.0
min_snr = 2.0
# -----

dt = np.median(np.diff(data["time"]))
sigma = np.mean(data["error"])
N = len(data["time"])

max_radius = star.min_radius(min_period, max_snr, N, sigma)
min_radius = star.min_radius(max_period, min_snr, N, sigma)

period_radius = np.array(
    np.meshgrid(
        np.linspace(min_period, max_period, n), np.linspace(min_radius, max_radius, n)
    )
)
period_radius = period_radius.reshape(2, n * n)
output = snakemake.output[0]
pickle.dump(period_radius, open(output, "wb"))
