import yaml
import numpy as np
import yaml
from astropy.timeseries import BoxLeastSquares
from time import time as _time
from wotan import flatten

times = {}

n_durations = int(eval(snakemake.wildcards.durations.split("_")[-1]))
periods = np.linspace(*[eval(i) for i in snakemake.wildcards.periods.split("_")[1:]])
durations = np.linspace(0.01, 0.05, n_durations)
time, flux, error = np.load(snakemake.input[0])
n_points = int(snakemake.wildcards.n_points)

t0 = _time()

flatten_trend = flatten(
    time, flux, window_length=0.2, return_trend=True
)[1]
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0
time = _time() - t0
times["biweight"] = float(time)

model = BoxLeastSquares(time, flatten_flux, dy=error)
results = model.power(periods, durations)#, objective="snr")
time = _time() - t0
times["bls"] = float(time)
yaml.safe_dump(times, open(snakemake.output[0], "w"))
