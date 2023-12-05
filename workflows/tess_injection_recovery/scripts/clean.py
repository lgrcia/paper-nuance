import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

i = 0

data = pickle.load(open(snakemake.input.fluxes, "rb"))
config = snakemake.config["cleaning"]

mask = data["time"] < data["time"].mean()
n = config["trimming"]
data["time"] = data["time"][mask][n:-n]
data["flux"] = data["flux"][mask][n:-n]
data["error"] = data["error"][mask][n:-n]

info = yaml.safe_load(open(snakemake.input.info, "r"))
gp_params = yaml.full_load(open(snakemake.input.gp, "r"))

build_gp, init = rotation(info["star_period"], np.mean(data["error"]), long_scale=0.5)
gp = build_gp(gp_params, data["time"])
nu = Nuance(data["time"], data["flux"], gp=gp)

cleaned_nu = nu.mask_flares(build_gp=build_gp, init=init, sigma=config["sigma"], iterations=config["iterations"])

cleaned_data = data.copy()
cleaned_data["flux"] = cleaned_nu.flux
cleaned_data["time"] = cleaned_nu.time
cleaned_data["error"] = np.ones_like(cleaned_nu.flux) * np.mean(data["error"])

pickle.dump(cleaned_data, open(snakemake.output[0], "wb"))
