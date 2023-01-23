import numpy as np
import yaml
import tinygp
from nuance import Nuance
from time import time as ctime
import jax 
jax.config.update("jax_enable_x64", True)


def search(time, flux, error, params, verbose=False):
    # the right one
    kernel = tinygp.kernels.quasisep.SHO(params['omega'], params['quality'], sigma=params["sigma"])
    
    nu = Nuance(time, flux, params['error'], kernel=kernel, mean=np.median(flux))
    t0s = time.copy()

    Ds = np.linspace(0.01, 0.1, 8)
    nu.linear_search(t0s, Ds, progress=verbose)

    periods = np.linspace(1, 1.8, 1000)
    search_data = nu.periodic_search(periods, progress=verbose)

    t0, _, P = search_data.best
    
    return t0, P, periods, search_data.Q_snr

if __name__=="__main__":
    
    seed = int(snakemake.wildcards.seed)
    time, flux, error = np.load(snakemake.input[0])
    params = yaml.full_load(open(snakemake.input[1], "r"))

    ct0 = ctime()
    t0, p0, _, _ = search(time, flux, error, params)
    t = ctime() - ct0

    result = {
        "t0": float(t0),
        "period": float(p0),
        "time": float(t)
    }

    yaml.safe_dump(result, open(snakemake.output[0], "w"))