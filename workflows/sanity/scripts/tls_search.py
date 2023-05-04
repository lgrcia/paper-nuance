import sys

import numpy as np

sys.path.append("../lib")
import yaml
from tls import tls

time, _, error = np.load(snakemake.input.lc)
flux = np.load(snakemake.input.injected)
period_range = snakemake.config["period_range"]
n_periods = snakemake.config["n_periods"]
duration = snakemake.config["duration"]
params = np.load(snakemake.input.params)
i = int(snakemake.wildcards.i)
periods = np.linspace(-0.1 + period_range[0], 0.1 + period_range[1], n_periods)

model = tls(time, flux, verbose=False)
results = model.power(
    periods,
    verbose=False,
    use_threads=1,
    show_progress_bar=False,
    durations=[0.01, duration],
)

t0, period, power = results["T0"], results["period"], results["power"]
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
