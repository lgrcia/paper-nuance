import pickle

import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

lc_file, gp_file, periods_file = snakemake.input
periods = np.load(periods_file)
data = pickle.load(open(lc_file, "rb"))
gp_params = yaml.full_load(open(gp_file, "r"))
build_gp, _ = rotation(data["star_period"])
gp = build_gp(gp_params, data["time"])

nu = Nuance(data["time"], data["flux"], gp=gp)

nu.linear_search(data["time"], np.array([0.01, data["transit_duration"]]))

search = nu.periodic_search(periods)

output = snakemake.output[0]
result = data.copy()
del result["flux"]
del result["error"]
del result["time"]
result.update(dict(zip(["found_t0", "found_duration", "found_period"], search.best)))
yaml.safe_dump(
    {name: float(value) for name, value in result.items()}, open(output, "w")
)
