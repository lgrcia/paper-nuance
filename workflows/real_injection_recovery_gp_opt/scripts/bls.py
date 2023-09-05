import sys

import numpy as np

sys.path.append("lib")
import pickle

import numpy as np
import yaml
from astropy.timeseries import BoxLeastSquares
from scipy.interpolate import BSpline, splrep
from wotan import flatten

data = pickle.load(open(snakemake.input.fluxes, "rb"))
periods = np.load(snakemake.input.periods)
info = yaml.safe_load(open(snakemake.input.info, "r"))

time, flux, error = data["time"], data["flux"], data["error"]
verbose = True

# biweight
# --------

flatten_trend = flatten(
    time, flux, window_length=data["transit_duration"] * 3, return_trend=True
)[1]
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

model = BoxLeastSquares(time, flatten_flux, dy=error)
results = model.power(periods, 0.03)

t0 = 0.0
period = results.period[np.argmax(results.power)]
power = results.power

pickle.dump(
    {"t0": t0, "period": period, "power": power, "trend": flatten_trend},
    open(snakemake.output.wotan3D, "wb"),
)


# harmonics
# ---------


# makes an harmonics design matrix of time
def make_harmonics(time, period, nharmonics=4):
    # make design matrix
    X = np.ones((len(time), 2 * nharmonics + 1))
    X[:, 1] = time
    for i in range(1, nharmonics):
        X[:, 2 * i + 1] = np.sin(2 * np.pi * i * time / period)
        X[:, 2 * i + 2] = np.cos(2 * np.pi * i * time / period)

    return X


X = make_harmonics(time, info["star_period"], nharmonics=2)
# solve for the coefficients
coeffs = np.linalg.solve(X.T @ X, X.T @ flux)
# make the model
flatten_trend = X @ coeffs
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

model = BoxLeastSquares(time, flatten_flux, dy=error)
results = model.power(periods, 0.03)

t0 = 0.0
period = results.period[np.argmax(results.power)]
power = results.power

pickle.dump(
    {"t0": t0, "period": period, "power": power, "trend": flatten_trend},
    open(snakemake.output.harmonics, "wb"),
)

# bspline
mask = np.ones_like(data["time"], dtype=bool)

for i in range(2):
    tck = splrep(
        time[mask],
        flux[mask],
        w=1 / error[mask],
    )
    trend = BSpline(*tck)(time)
    mask &= np.abs(flux - trend) < 3 * np.std(flux - trend)

trend = BSpline(*tck)(time)

flatten_trend = BSpline(*tck)(time)
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

model = BoxLeastSquares(time, flatten_flux, dy=error)
results = model.power(periods, 0.03)

t0 = 0.0
period = results.period[np.argmax(results.power)]
power = results.power

pickle.dump(
    {"t0": t0, "period": period, "power": power, "trend": flatten_trend},
    open(snakemake.output.bspline, "wb"),
)
