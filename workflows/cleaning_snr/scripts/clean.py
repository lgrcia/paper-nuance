import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import yaml
from wotan import flatten

params = yaml.full_load(open(snakemake.input[0], "r"))
time, flux, error = np.load(snakemake.input[1])

intransit = np.abs(time - params['t0']) < params['duration']
n_tr = params['duration']/np.median(np.diff(time))

flatten_flux = flatten(time, flux + 1., window_length=3*params["duration"], return_trend=False)
new_depth = np.max([0, np.median(flatten_flux[~intransit]) - np.median(flatten_flux[intransit])])
w = params["error"]
r = np.max([0, np.std(flatten_flux[~intransit])-w])
snr = new_depth/np.sqrt((w**2)/n_tr + r**2)
result = {
    "relative_duration" : float(params["relative_duration"]),
    "relative_depth" : float(params["relative_depth"]),
    "snr": float(snr)
}
yaml.safe_dump(result, open(snakemake.output[0], "w"))