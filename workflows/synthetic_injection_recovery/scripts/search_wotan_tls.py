import numpy as np
import yaml
from transitleastsquares import transitleastsquares as TLS
from wotan import flatten
from time import time as ctime

seed = int(snakemake.wildcards.seed)

time, flux, error = np.load(snakemake.input[0])
params = yaml.full_load(open(snakemake.input[1], "r"))
flux += 1

t0 = ctime()

flatten_flux, flatten_trend = flatten(time, flux, window_length=3*params['duration'], return_trend=True)

model = TLS(time, flatten_flux, verbose=False)
results = model.power(verbose=False, use_threads=1, show_progress_bar=False)

t = ctime() - t0

result = {
    "t0": float(results["T0"]),
    "period": float(results["period"]),
    "time": float(t),
}


yaml.safe_dump(result, open(snakemake.output[0], "w"))