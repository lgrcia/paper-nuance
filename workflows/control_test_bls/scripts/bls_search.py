import numpy as np
import yaml
from astropy.timeseries import BoxLeastSquares

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
model = BoxLeastSquares(time, flux, dy=error)
results = model.power(periods, duration)

yaml.safe_dump(
    {
        "period": float(results.period[np.argmax(results.power)]),
    },
    open(snakemake.output[0], "w"),
)
