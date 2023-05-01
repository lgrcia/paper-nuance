import pickle

import numpy as np
from nuance.star import Star

input = snakemake.input[0]
data = pickle.load(open(input, "rb"))
n = snakemake.params["n"]

star = Star(
    data["star_radius"], data["star_mass"], data["star_amplitude"], data["star_period"]
)

min_period = 0.2
max_period = 10.0
max_snr = 30.0
min_snr = 4.0
# -----

dt = np.median(np.diff(data["time"]))
sigma = np.mean(data["error"])
radii = np.linspace(0.1, 50, 100000)
N = len(data["time"])

max_radius = star.min_radius(max_period, max_snr, N, sigma)
min_radius = star.min_radius(max_period, min_snr, N, sigma)

period_radius = np.array(
    np.meshgrid(
        np.linspace(min_period, max_period, n), np.linspace(min_radius, max_radius, n)
    )
)
period_radius = period_radius.reshape(2, n * n)
output = snakemake.output[0]
pickle.dump(period_radius, open(output, "wb"))
