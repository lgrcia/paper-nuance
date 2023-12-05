from glob import glob
import pandas as pd
import yaml
import pickle
import numpy as np
from nuance.star import Star

method = "bls_bspline"
snr_limit = 6

info = yaml.safe_load(open(snakemake.input[0], "r"))
star = Star(info["star_radius"], info["star_mass"])
df = pd.read_csv(snakemake.input[1])
cleaned = pickle.load(open(snakemake.input[2], "rb"))

true_snr = df.apply(
    lambda row: star.snr(
        row["true_period"],
        row["radius"],
        len(cleaned["flux"]),
        np.median(cleaned["error"]),
    ),
    axis=1,
)


def right_candidate(period, true_period):
    period_check = np.abs(period - true_period) <= 0.01
    alias_check = np.abs(2 * period - true_period) <= 0.01
    alias_check |= np.abs(period / 2 - true_period) <= 0.01
    result = np.zeros_like(period)
    result[period_check] = 1
    result[alias_check] = 0.5
    return result


snr = df["snr"]
df["true_snr"] = true_snr
period_ok = right_candidate(df["period"], df["true_period"])
detectable = (df["true_snr"] > snr_limit).values

true_positives = np.logical_and(period_ok > 0.0, np.logical_and(detectable, df["snr"].values >= snr_limit))
false_positives = np.logical_and(~detectable, df["snr"] > snr_limit)
tau, delta = df["tau"].values, df["delta"].values
result = np.vstack([tau, delta, true_positives, false_positives, detectable])

np.save(snakemake.output[0], result)