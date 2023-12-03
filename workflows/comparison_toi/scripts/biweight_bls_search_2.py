import numpy as np
from astropy.timeseries import BoxLeastSquares
from nuance.utils import phase
from time import time as timef
import yaml

periods, durations = snakemake.params[0:2]

t = timef()


def bls(time, flux, error):
    model = BoxLeastSquares(time, flux, dy=error)
    return model.power(periods, durations, objective="snr")


time, flux, error = np.load(snakemake.input[0])
t0, D, P = np.load(snakemake.input[1])

mask = phase(time, t0, P) >= 2 * D

results = bls(time[mask], flux[mask], error[mask])

t = float(timef() - t)

yaml.safe_dump({"time": t}, open(snakemake.output[1], "w"))
pickle.dump(results, open(snakemake.output[0], "wb"))
