# make_lightcurve.py
import jax
import numpy as np
import tinygp
import yaml
from nuance.utils import periodic_transit

jax.config.update("jax_enable_x64", True)

params = yaml.full_load(open(snakemake.input[0], "r"))
i = int(snakemake.wildcards.i)

time = np.arange(0, 4, 2 / 60 / 24)

kernel = tinygp.kernels.quasisep.SHO(
    params["omega"], params["quality"], sigma=params["sigma"]
)

gp = tinygp.GaussianProcess(kernel, time, diag=params["error"] ** 2)

transit = params["depth"] * periodic_transit(
    time, params["t0"], params["duration"], P=params["period"], c=500
)
flux = gp.sample(jax.random.PRNGKey(i)) + transit + 1.0
error = np.ones_like(flux) * params["error"]

np.save(snakemake.output[0], np.array([time, flux, error]))
