import lightkurve as lk
import pandas as pd
import yaml

tic = int(snakemake.wildcards.target)
info_output = snakemake.output[0]

targets = pd.read_csv("data/tess-ultra-fast-rotators-brighter-mag14.csv")[
    ["Name", "LS_Period", "Amplitude", "teff_val"]
]

targets = targets.sort_values("LS_Period")
target = targets[targets.Name == tic]

name = f"TIC {tic}"
search_result = lk.search_lightcurve(name, author="SPOC", exptime=120)

# sector
sector = search_result[0].mission[0].split(" ")[-1]
sector = int(sector.lstrip("0"))

# first exposure
first_exposure = search_result[0].download().time[0]
time_format = rf"{first_exposure.format.upper()}_{{{first_exposure.scale.upper()}}}"
first_exposure_time = first_exposure.value

info = {
    "tic": int(tic),
    "sector": int(sector),
    "ramsay_teff": float(target.teff_val.values[0]),
    "ramsay_period": float(target.LS_Period.values[0]),
    "first_exposure_time": float(first_exposure_time),
    "first_exposure_time_format": time_format,
}

yaml.safe_dump(info, open(info_output, "w"))
