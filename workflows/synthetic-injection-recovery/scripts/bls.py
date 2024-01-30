import numpy as np
from astropy.timeseries import BoxLeastSquares
import pickle

time, flatten_flux, error = np.load(snakemake.input.lc)

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
