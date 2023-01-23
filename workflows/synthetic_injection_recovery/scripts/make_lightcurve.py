# make_lightcurve.py
import tinygp
import jax
import numpy as np
import yaml
from nuance.utils import periodic_transit
jax.config.update("jax_enable_x64", True)

def make_lc(params, seed):
    time = np.arange(0, 3, 2/60/24)
    
    kernel = tinygp.kernels.quasisep.SHO(params['omega'], params['quality'], sigma=params["sigma"])
    gp = tinygp.GaussianProcess(kernel, time, diag=params['error']**2)

    transit = params["depth"]*periodic_transit(time, params["t0"], params["duration"], P=params["period"])
    y = gp.sample(jax.random.PRNGKey(seed)) + transit
    
    error = np.ones_like(y)*params['error']
    
    return time, y, error

if __name__=="__main__":
    
    seed = int(snakemake.wildcards.seed)
    params = yaml.full_load(open(snakemake.input[0], "r"))
    time, y, error = make_lc(params, seed)
    np.save(snakemake.output[0], np.array([time, y, error]))