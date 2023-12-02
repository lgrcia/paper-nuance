import pickle

import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd
import yaml

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
times = klc.time.to_value("btjd")
fluxes = klc.pdcsap_flux.to_value().filled(np.nan)
errors = klc.flux_err.to_value().filled(np.nan)
mask = np.isnan(fluxes) | np.isnan(errors) | np.isnan(times)
original_time = times[~mask]
original_flux = fluxes[~mask]
original_error = errors[~mask]

splits = np.array_split(
    np.arange(len(original_time)), np.flatnonzero(np.diff(original_time) > 0.1) + 1
)

n = 300
time = np.hstack([original_time[split[n:-n]] for split in splits])
flux = np.hstack([original_flux[split[n:-n]] for split in splits])
error = np.hstack([original_error[split[n:-n]] for split in splits])

# first exposure
first_exposure = klc.time[0]
time_format = rf"{first_exposure.format.upper()}_{{{first_exposure.scale.upper()}}}"
first_exposure_time = first_exposure.value

flux_median = np.median(original_flux)
time = time - first_exposure_time
flux /= flux_median
error /= flux_median

# stellar parameters
url = f"https://exofop.ipac.caltech.edu/tess/download_stellar.php?id={tic}"
star = pd.read_csv(url, delimiter="|", index_col=1).iloc[0]

pickle.dump(
    {
        "flux": flux,
        "time": time,
        "error": error,
    },
    open(snakemake.output.fluxes, "wb"),
)


star_dict = {
    "tic": int(tic),
    "time0": float(first_exposure_time),
    "time0_format": str(time_format),
    "star_period": float(target.LS_Period),
    "star_amplitude": float(target.Amplitude),
    "star_radius": float(star["Radius (R_Sun)"]),
    "star_mass": float(star["Mass (M_Sun)"]),
    "star_teff": float(star["Teff (K)"]),
    "star_logg": float(star["log(g)"]),
    "sector": int(klc.sector),
    "logg": float(star["log(g)"]),
}

ld = pd.read_csv("data/tess_limb_darkening.txt", delim_whitespace=True, comment="#")
closest_teff = ld.Teff.iloc[np.argmin(np.abs(ld.Teff - star_dict["star_teff"]))]
teff_ld = ld[ld.Teff == closest_teff]
ld_row = teff_ld.iloc[np.argmin(np.abs(teff_ld.logg - star_dict["star_logg"]))]
ld_coeffs = ld_row[["aLSM", "bLSM"]].values.astype(float)
float(ld_coeffs[0]), float(ld_coeffs[1])
star_dict["ld_coeffs"] = float(ld_coeffs[0]), float(ld_coeffs[1])

yaml.safe_dump(
    star_dict,
    open(snakemake.output.info, "w"),
)

print(f"Downloaded TIC {tic} data (first sector)")
