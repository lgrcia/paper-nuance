import sys

import numpy as np

sys.path.append("lib")
import pickle

import numpy as np
import yaml
from tls import tls
from wotan import flatten

period_range = snakemake.params.period_range
time, flux, _ = np.load(snakemake.input[0])
n_points = int(snakemake.wildcards.n_points)
n_X = int(snakemake.wildcards.n_X)
n_periods = int(snakemake.wildcards.n_periods)
periods = np.linspace(period_range[0], period_range[1], n_periods)

verbose = True

flatten_trend = flatten(time, flux, window_length=0.01, return_trend=True)[1]
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

model = tls(time, flatten_flux, verbose=verbose)
results = model.power(
    periods,
    verbose=verbose,
    use_threads=1,
    show_progress_bar=verbose,
    durations=[0.01, 0.02],
)
