import numpy as np
import yaml
import sys
sys.path.append("./lib")
sys.path.append("../lib")
from tls import tls
from wotan import flatten
from time import time as ctime

t0 = ctime()

def trend(time, flux, params):
    return flatten(time, flux+1000., window_length=3*params['duration'], return_trend=True)[1] - 1000.

def search(time, flux, error, params, verbose=False):
    flatten_trend = trend(time, flux, params)
    flatten_flux = flux - flatten_trend
    flatten_flux -= np.mean(flatten_flux)
    flatten_flux += 1.

    model = tls(time, flatten_flux, verbose=verbose)
    periods = np.linspace(1, 1.8, 1000)
    results = model.power(periods, verbose=verbose, use_threads=1, show_progress_bar=verbose)
    
    return results["T0"], results["period"], periods, results["power"]


if __name__=="__main__":
    
    seed = int(snakemake.wildcards.seed)
    time, flux, error = np.load(snakemake.input[0])
    params = yaml.full_load(open(snakemake.input[1], "r"))
    
    ct0 = ctime()
    t0, p0, periods, periodogram = search(time, flux, error, params)
    t = ctime() - ct0

    result = {
        "t0": float(t0),
        "period": float(p0),
        "time": float(t)
    }

    yaml.safe_dump(result, open(snakemake.output[0], "w"))