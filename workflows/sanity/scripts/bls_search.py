import numpy as np
import yaml
from astropy.timeseries import BoxLeastSquares

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
periods = np.linspace(0.5, 6, 3000)
model = BoxLeastSquares(time, flux, dy=error)
results = model.power(periods, 0.03)

yaml.safe_dump(
    {
        "t0": float(0.0),
        "period": float(results.period[np.argmax(results.power)]),
        "true_period": float(params[i, 0]),
        "true_t0": float(0),
        "true_depth": float(params[i, 1]),
    },
    open(snakemake.output[0], "w"),
)
