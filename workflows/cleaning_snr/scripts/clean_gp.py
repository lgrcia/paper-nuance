import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import yaml
from utils import snr
from make_lightcurve import build_gp

params = yaml.full_load(open(snakemake.input[0], "r"))
time, flux, error = np.load(snakemake.input[1])
gp = build_gp(time, params)
_, cond = gp.condition(y=flux, X_test=time)
flatten_flux = flux - cond.mean

result = {
    "relative_duration" : float(params["relative_duration"]),
    "relative_depth" : float(params["relative_depth"]),
    "snr": float(snr(time, flatten_flux, params["t0"], params["duration"], params["error"])),
}
yaml.safe_dump(result, open(snakemake.output[0], "w"))