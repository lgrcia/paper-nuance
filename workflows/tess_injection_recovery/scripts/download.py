import pickle

import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd
import yaml

tic = int(snakemake.wildcards["target"])
filepath = snakemake.output[0]

targets = pd.read_csv("static/tess-ultra-fast-rotators-brighter-mag14-clean.csv")

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
time = times[~mask]
flux = fluxes[~mask]
error = errors[~mask]

# first exposure
first_exposure = klc.time[0]
time_format = rf"{first_exposure.format.upper()}_{{{first_exposure.scale.upper()}}}"
first_exposure_time = first_exposure.value

flux_median = np.median(flux)
time = time - first_exposure_time
flux /= flux_median
error /= flux_median

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
    "star_radius": float(target.star_radius),
    "star_mass": float(target.star_mass),
    "star_logg": float(target.star_logg),
    "sector": int(klc.sector),
}

# ld = pd.read_csv("static/tess_limb_darkening.txt", delim_whitespace=True, comment="#")
# closest_teff = ld.Teff.iloc[np.argmin(np.abs(ld.Teff - star_dict["star_teff"]))]
# teff_ld = ld[ld.Teff == closest_teff]
# ld_row = teff_ld.iloc[np.argmin(np.abs(teff_ld.logg - star_dict["star_logg"]))]
# ld_coeffs = ld_row[["aLSM", "bLSM"]].values.astype(float)
# float(ld_coeffs[0]), float(ld_coeffs[1])
# star_dict["ld_coeffs"] = float(ld_coeffs[0]), float(ld_coeffs[1])

yaml.safe_dump(
    star_dict,
    open(snakemake.output.info, "w"),
)

print(f"Downloaded TIC {tic} data (first sector)")
