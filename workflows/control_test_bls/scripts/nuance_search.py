import numpy as np
import yaml
from nuance import Nuance

time, _, error = np.load(snakemake.input.lc)
flux = np.load(snakemake.input.injected)
duration = snakemake.config["duration"]
params = yaml.full_load(open(snakemake.input.params))
period_range = snakemake.config["period_range"]
periods = np.linspace(
    -0.1 + period_range[0], 0.1 + period_range[1], snakemake.config["n_periods"]
)

# search
# ------
nu = Nuance(time, flux, error)
nu.linear_search(time, np.array([0.01, duration]))
search = nu.periodic_search(periods)
t0, _, period = search.best

yaml.safe_dump(
    {
        "t0": float(t0),
        "period": float(period),
    },
    open(snakemake.output[0], "w"),
)
