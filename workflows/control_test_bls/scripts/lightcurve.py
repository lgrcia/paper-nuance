import numpy as np

np.random.seed(0)
# 2 min cadence
time = np.arange(0, 6, 2 / 60 / 24)
error = np.ones_like(time) * 0.0005
flux = np.random.normal(loc=1.0, scale=error[0], size=len(time))

np.save(
    snakemake.output[0],
    np.array([time, flux, error]),
)
