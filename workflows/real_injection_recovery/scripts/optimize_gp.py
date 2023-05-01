import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

input = snakemake.input[0]
data = pickle.load(open(input, "rb"))

# only one continuous segment
dt = np.diff(data["time"])
mask = data["time"] < data["time"][np.flatnonzero(dt > 0.5)[0]]
data["flux"] = data["flux"][mask]
data["time"] = data["time"][mask]
data["error"] = data["error"][mask]
# sigma clip
mask = np.abs(data["flux"] - np.median(data["flux"])) < 3 * np.std(data["flux"])
data["flux"] = data["flux"][mask]
data["time"] = data["time"][mask]
data["error"] = data["error"][mask]

build_gp, init = rotation(data["star_period"], np.mean(data["error"]), long_scale=0.5)
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
