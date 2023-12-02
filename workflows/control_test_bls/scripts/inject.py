import numpy as np
import yaml
from nuance.utils import periodic_transit

time, flux, error = np.load(snakemake.input.lc)
params = yaml.full_load(open(snakemake.input.params))
duration = snakemake.config["duration"]
i = int(snakemake.wildcards.i)

np.random.seed(i)
period = params["period"]
t0 = params["t0"]
depth = params["depth"]
# c = 500 to have a box shaped transit
injected_flux = flux + depth * periodic_transit(time, t0, duration, period, c=500)

np.save(snakemake.output[0], injected_flux)
