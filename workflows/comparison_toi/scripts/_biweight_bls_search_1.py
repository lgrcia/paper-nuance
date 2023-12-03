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

# TOI-540 tranis tis 0.48 hrs
window = 0.02
flatten_trend = np.hstack(
    [
        flatten(
            n.time,
            n.flux,
            window_length=3 * window,
            return_trend=True,
            robust=True,
        )[1]
        for n in nus
    ]
)

import matplotlib.pyplot as plt

plt.plot(nu_s.time, nu_s.flux)
plt.plot(nu_s.time, flatten_trend)
plt.savefig("test.png")

flatten_flux = nu_s.flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

error = np.median([np.median(np.sqrt(n.gp.variance)) for n in nus]) * np.ones_like(
    flatten_flux
)

results = bls(nu_s.time, flatten_flux, error)

t = float(timef() - t)

pickle.dump(results, open(snakemake.output[0], "wb"))
np.save(snakemake.output[1], [nu_s.time, flatten_flux, error])
yaml.safe_dump({"time": t}, open(snakemake.output[2], "w"))
