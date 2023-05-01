import sys

import numpy as np

sys.path.append("lib")
import pickle

import numpy as np
import yaml
from scipy.interpolate import BSpline, splrep
from tls import tls

data = pickle.load(open(snakemake.input.fluxes, "rb"))
periods = np.load(snakemake.input.periods)
info = yaml.safe_load(open(snakemake.input.info, "r"))
verbose = True
mask = np.ones_like(data["time"], dtype=bool)
time, flux, error = data["time"], data["flux"], data["error"]

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

model = tls(time, flatten_flux, verbose=verbose)
results = model.power(
    periods,
    verbose=verbose,
    use_threads=1,
    show_progress_bar=verbose,
    durations=[0.01, data["transit_duration"]],
)

t0, period, power = results["T0"], results["period"], results["power"]
pickle.dump(
    {"t0": t0, "period": period, "power": power, "trend": flatten_trend},
    open(snakemake.output[0], "wb"),
)
