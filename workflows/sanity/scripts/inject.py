import numpy as np
from nuance.utils import periodic_transit

time, flux, error = np.load(snakemake.input.lc)
params = np.load(snakemake.input.params)
i = int(snakemake.wildcards.i)
duration = snakemake.config["duration"]

np.random.seed(i)
period, depth = params[i]
injected_flux = flux + depth * periodic_transit(time, 0, duration, period)

np.save(snakemake.output[0], injected_flux)
