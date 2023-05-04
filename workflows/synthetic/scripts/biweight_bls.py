import numpy as np
import yaml
from astropy.timeseries import BoxLeastSquares
from wotan import flatten

time, flux, error = np.load(snakemake.input.lc)

flatten_trend = flatten(
    time, flux, window_length=snakemake.config["duration"] * 3, return_trend=True
)[1]
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

# search
# ------
periods = np.linspace(0.8, 3, 1000)
model = BoxLeastSquares(time, flatten_flux, dy=error)
results = model.power(periods, snakemake.config["duration"])

pickle.dump(
    {
        "t0": float(0.0),
        "period": float(results.period[np.argmax(results.power)]),
        "power": results.power,
    },
    open(snakemake.output[0], "wb"),
)
