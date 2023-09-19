import pickle

import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation
import matplotlib.pyplot as plt
from nuance.utils import plot_search

gp_params = yaml.full_load(open(snakemake.input.gp, "r"))
data = pickle.load(open(snakemake.input.fluxes, "rb"))
periods = np.load(snakemake.input.periods)
info = yaml.safe_load(open(snakemake.input.info, "r"))

build_gp, _ = rotation()

gp = build_gp(gp_params, data["time"])
nu = Nuance(data["time"], data["flux"], gp=gp)

Ds = np.linspace(0.01, 0.03, 5)
# Ds = np.array([0.01, data["transit_duration"]])
nu.linear_search(data["time"], Ds)

search = nu.periodic_search(periods)

plt.figure(figsize=(15, 5))
plot_search(nu, search)
plt.savefig(snakemake.output[0])
