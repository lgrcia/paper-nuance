import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

data = pickle.load(open(snakemake.input.fluxes, "rb"))
info = yaml.safe_load(open(snakemake.input.info, "r"))

# only one continuous segment (removed later so commented)
dt = np.diff(data["time"])
mask = np.ones_like(
    data["time"], dtype=bool
)  # < data["time"][np.flatnonzero(dt > 0.5)[0]]
data["flux"] = data["flux"][mask]
data["time"] = data["time"][mask]
data["error"] = data["error"][mask]
# sigma clip
mask = np.abs(data["flux"] - np.median(data["flux"])) < 3 * np.std(data["flux"])
data["flux"] = data["flux"][mask]
data["time"] = data["time"][mask]
data["error"] = data["error"][mask]

build_gp, init = rotation(info["star_period"], np.mean(data["error"]), long_scale=0.5)
nu = Nuance(data["time"], data["flux"], data["error"])
optimize, mu, nll = nu.gp_optimization(build_gp)

new = optimize(
    init, ["log_sigma", "log_short_scale", "log_short_sigma", "log_long_sigma"]
)
new = optimize(new)

gp_output = snakemake.output[0]
yaml.safe_dump(
    {name: float(value) for name, value in new.items()}, open(gp_output, "w")
)
