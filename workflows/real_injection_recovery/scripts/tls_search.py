import sys

import numpy as np
from nuance.utils import clean_periods

sys.path.append("lib")
import pickle

import numpy as np
import yaml
from tls import tls
from wotan import flatten

lc_file = snakemake.input[0]
periods_file = snakemake.input[1]
periods = np.load(periods_file)
data = pickle.load(open(lc_file, "rb"))


def trend(time, flux, window_length):
    return (
        flatten(time, flux + 1000.0, window_length=window_length, return_trend=True)[1]
        - 1000.0
    )


def search(time, flux, window_length, periods, verbose=False):
    flatten_trend = trend(time, flux, window_length)
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

    return results["T0"], results["period"]


t0, period = search(
    data["time"], data["flux"], 3 * data["transit_duration"], periods, verbose=True
)

output = snakemake.output[0]
result = data.copy()
del result["flux"]
del result["error"]
del result["time"]
result.update(
    dict(zip(["found_t0", "found_duration", "found_period"], [t0, -1, period]))
)
yaml.safe_dump(
    {name: float(value) for name, value in result.items()}, open(output, "w")
)
