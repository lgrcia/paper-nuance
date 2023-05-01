import pickle

import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd

tic = int(snakemake.wildcards["target"])
filepath = snakemake.output[0]

targets = pd.read_csv("data/tess-ultra-fast-rotators-brighter-mag14.csv")[
    ["Name", "LS_Period", "Amplitude", "teff_val"]
]

targets = targets.sort_values("LS_Period")
target = targets[targets.Name == tic]

name = f"TIC {tic}"
search_result = lk.search_lightcurve(name, author="SPOC", exptime=120)

# data
klc = search_result[0].download()
times = klc.time.to_value("mjd")
fluxes = klc.pdcsap_flux.to_value().filled(np.nan)
errors = klc.flux_err.to_value().filled(np.nan)
masks = [
    np.isnan(f) | np.isnan(e) | np.isnan(t) for f, e, t in zip(fluxes, errors, times)
]
times = [t[~m] for t, m in zip(times, masks)]

# masking
time = np.hstack(times)
time -= np.min(time)
original_flux = np.hstack([f[~m] for f, m in zip(fluxes, masks)])
original_error = np.hstack([e[~m] for e, m in zip(errors, masks)])

flux_median = np.median(original_flux)
flux = original_flux / flux_median
error = original_error / flux_median

# stellar parameters
url = f"https://exofop.ipac.caltech.edu/tess/download_stellar.php?id={tic}"
star = pd.read_csv(url, delimiter="|", index_col=1).iloc[0]

pickle.dump(
    {
        "flux": flux,
        "time": time,
        "error": error,
        "star_period": float(target.LS_Period),
        "star_amplitude": float(target.Amplitude),
        "star_radius": star["Radius (R_Sun)"],
        "star_mass": star["Mass (M_Sun)"],
    },
    open(filepath, "wb"),
)
print(f"Downloaded TIC {tic} data (first sector)")
