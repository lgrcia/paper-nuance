import pickle
from nuance.combined import CombinedNuance

periods, durations = snakemake.params
nus = [pickle.load(open(f, "rb")) for f in snakemake.input]
nu_s = CombinedNuance(nus)
len(nu_s.time)

import numpy as np
from astropy.timeseries import BoxLeastSquares
from wotan import flatten
from tqdm import tqdm
from time import time as timef
import yaml


def bls(time, flux, error):
    model = BoxLeastSquares(time, flux, dy=error)
    return model.power(periods, durations, objective="snr")


verbose = True

# biweight
# --------
t = timef()

# This was the sherlock-like approach
windows = np.linspace(30 / 60 / 24, 5 / 24, 15)
results_ = []
flatten_fluxes = []

for window in tqdm(windows):
    flatten_trend = np.hstack(
        [
            flatten(
                n.time,
                n.flux,
                window_length=window,
                return_trend=True,
                robust=True,
            )[1]
            for n in nus
        ]
    )

    flatten_flux = nu_s.flux - flatten_trend
    flatten_flux -= np.mean(flatten_flux)
    flatten_flux += 1.0

    error = np.median([np.median(np.sqrt(n.gp.variance)) for n in nus]) * np.ones_like(
        flatten_flux
    )

    results = bls(nu_s.time, flatten_flux, error)
    results_.append(results)
    flatten_fluxes.append(flatten_flux)

t = float(timef() - t)

i = np.argmax([np.max(r.power) for r in results_])

pickle.dump(results_[i], open(snakemake.output[0], "wb"))
np.save(snakemake.output[1], [nu_s.time, flatten_fluxes[i], error])
yaml.safe_dump({"time": t}, open(snakemake.output[2], "w"))
