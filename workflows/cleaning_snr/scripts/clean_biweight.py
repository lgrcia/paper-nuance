import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import yaml
from utils import snr
from wotan import flatten

params = yaml.full_load(open(snakemake.input[0], "r"))
time, flux, error = np.load(snakemake.input[1])
flatten_flux = flatten(time, flux + 1., window_length=3*params["duration"], return_trend=False)

result = {
    "relative_duration" : float(params["relative_duration"]),
    "relative_depth" : float(params["relative_depth"]),
    "snr": float(snr(time, flatten_flux, params["t0"], params["duration"], params["error"])),
}
yaml.safe_dump(result, open(snakemake.output[0], "w"))