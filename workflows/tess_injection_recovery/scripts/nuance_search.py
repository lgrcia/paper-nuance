import pickle

import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

gp_params = yaml.full_load(open(snakemake.input.gp, "r"))
data = pickle.load(open(snakemake.input.fluxes, "rb"))
periods = np.load(snakemake.input.periods)
info = yaml.safe_load(open(snakemake.input.info, "r"))

build_gp, _ = rotation(info["star_period"])
gp = build_gp(gp_params, data["time"])

nu = Nuance(data["time"], data["flux"], gp=gp, c=500)

nu.linear_search(data["time"], np.array([0.01, data["transit_duration"]]))

search = nu.periodic_search(periods)

output = snakemake.output[0]
t0, _, period = search.best
pickle.dump(
    {"t0": t0, "period": period, "power": search.Q_snr, "trend": None},
    open(snakemake.output[0], "wb"),
)
