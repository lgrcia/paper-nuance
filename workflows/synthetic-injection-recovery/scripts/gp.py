import jax
import numpy as np
import tinygp
import yaml

jax.config.update("jax_enable_x64", True)

time, flux, error = np.load(snakemake.input.lc)
params = yaml.full_load(open(snakemake.input.params, "r"))

time = np.arange(0, 4, 2 / 60 / 24)

kernel = tinygp.kernels.quasisep.SHO(
    params["omega"], params["quality"], sigma=params["sigma"]
)

gp = tinygp.GaussianProcess(kernel, time, diag=params["error"] ** 2, mean=1.)

@jax.jit
def mu(y, X_test):
    _, cond =  gp.condition(y, X_test)
    return cond.mean

flatten_flux = flux - mu(flux, time)
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

np.save(snakemake.output[0], [time, flatten_flux, error])