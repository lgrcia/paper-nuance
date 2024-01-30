import numpy as np
from wotan import flatten

time, flux, error = np.load(snakemake.input.lc)

flatten_trend = flatten(
    time, flux, window_length=snakemake.config["duration"] * 3, return_trend=True
)[1]
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

np.save(snakemake.output[0], [time, flatten_flux, error])