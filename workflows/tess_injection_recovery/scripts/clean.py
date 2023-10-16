import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

i = 0

data = pickle.load(open(snakemake.input.fluxes, "rb"))

# splits = np.array_split(
#     np.arange(len(data["time"])), np.flatnonzero(np.diff(data["time"]) > 0.1) + 1
# )

mask = data["time"] < data["time"].mean()
n = 300
data["time"] = data["time"][mask][n:-n]
data["flux"] = data["flux"][mask][n:-n]
data["error"] = data["error"][mask][n:-n]

info = yaml.safe_load(open(snakemake.input.info, "r"))
gp_params = yaml.full_load(open(snakemake.input.gp, "r"))

build_gp, init = rotation(info["star_period"], np.mean(data["error"]), long_scale=0.5)
gp = build_gp(gp_params, data["time"])
nu = Nuance(data["time"], data["flux"], gp=gp)


def mask_flares(time, flux, iterations=3, sigma=3, gp=None):
    _, mu, _ = nu.gp_optimization(build_gp)

    mask = np.ones_like(time).astype(bool)
    window = 30

    for _ in range(iterations):
        m = mu(gp_params)
        r = flux - m
        mask_up = r < np.std(r[mask]) * sigma

        # mask around flares
        ups = np.flatnonzero(~mask_up)
        if len(ups) > 0:
            mask[
                np.hstack(
                    [
                        np.arange(max(u - window, 0), min(u + window, len(time)))
                        for u in ups
                    ]
                )
            ] = False
        _, mu, _ = nu.gp_optimization(build_gp, mask)

    return ~mask


mask = ~mask_flares(data["time"], data["flux"], gp=gp)

cleaned_data = data.copy()
cleaned_data["flux"] = data["flux"][mask]
cleaned_data["time"] = data["time"][mask]
cleaned_data["error"] = data["error"][mask]

pickle.dump(cleaned_data, open(snakemake.output[0], "wb"))
