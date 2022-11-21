# make_lightcurve.py
import tinygp
import jax
import numpy as np
import yaml
from nuance.utils import periodic_transit

seed = int(snakemake.wildcards.seed)

time = np.arange(0, 10, 2/60/24) + 0.23

params = yaml.full_load(open(snakemake.input[0], "r"))
kernel = tinygp.kernels.quasisep.SHO(params['omega'], params['quality'], sigma=params["sigma"])
gp = tinygp.GaussianProcess(kernel, time, diag=params['error']**2)

transit = periodic_transit(time, params["t0"], params["duration"], params["depth"], P=params["period"])
y = gp.sample(jax.random.PRNGKey(seed)) + transit

# plt.figure()
# plt.plot(time, y, "-")
# plt.plot(time, transit)

np.save(snakemake.output[0], np.array([time, y, np.ones_like(y)*params['error']]))