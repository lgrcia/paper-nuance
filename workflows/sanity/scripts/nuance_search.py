import numpy as np
import yaml
from nuance import Nuance

time, _, error = np.load(snakemake.input.lc)
flux = np.load(snakemake.input.injected)
period_range = snakemake.config["period_range"]
n_periods = snakemake.config["n_periods"]
duration = snakemake.config["duration"]
params = np.load(snakemake.input.params)
i = int(snakemake.wildcards.i)
periods = np.linspace(-0.1 + period_range[0], 0.1 + period_range[1], n_periods)

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
        "true_period": float(params[i, 0]),
        "true_t0": float(0),
        "true_depth": float(params[i, 1]),
    },
    open(snakemake.output[0], "w"),
)
