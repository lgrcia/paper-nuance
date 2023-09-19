# could not finish this one, but the idea is to plot the difference in detection rate

import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

methods = {
    "bls_bspline": "bspline + BLS",
    "bls_wotan3D": "biweight + BLS",
    "bls_harmonics": "harmonics + BLS",
    "bens": "sinusoids + BLS (Ben's)",
    "nuance": "nuance",
}


targets = [int(Path(f).stem) for f in glob("../figures/searched/*")]

result_df = pd.DataFrame(columns=targets)


def right_candidate(t0, period, true_t0, true_period, verbose=False):
    t0_check = (
        np.abs((t0 - true_t0 + 0.5 * true_period) % true_period - 0.5 * true_period)
        % period
        < 0.01
    )
    period_check = np.abs(period - true_period) <= 0.01
    alias_check = np.abs(2 * period - true_period) <= 0.01
    alias_check |= np.abs(period / 2 - true_period) <= 0.01
    if period_check:
        return 1
    elif alias_check:
        return 1
    else:
        return 0


for target in targets:
    for method in methods:
        df = pd.read_csv(
            f"../data/{target}/recovered/{method}/results.csv", index_col=0
        )
        df["found"] = df.apply(
            lambda row: right_candidate(0, row["period"], 0, row["true_period"]),
            axis=1,
        )
        total_found = np.count_nonzero(df.found) / len(df)

        result_df.loc[method, target] = total_found * 100

    result_df.loc["fresh_snr", target] = df.fresh_snr.values[0]


gp_characteristics = {}

import yaml

for target in targets:
    gp_characteristics[target] = yaml.safe_load(open(f"../data/{target}/gp.yaml"))

gp_characteristics = pd.DataFrame(gp_characteristics)


import matplotlib.pyplot as plt

snr = 6

# exclude nuance
good_ones = result_df.loc[:, (result_df.loc["fresh_snr"] < snr).values]
difference = good_ones.iloc[-2] - good_ones.drop("nuance").max(axis=0)
gp_char = gp_characteristics[good_ones.columns]

# grid of plots

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

n = int(np.ceil(np.sqrt(len(gp_char.index))))

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(n, n, figure=fig)

for i, key in enumerate(gp_char.index):
    if i < len(gp_char.index):
        ax = fig.add_subplot(gs[i])
        plt.plot(gp_char.loc[key].values, difference, ".")
        plt.axhline(0, c="0.8")
        mean = np.median(gp_char.loc[key].values)
        std = np.std(gp_char.loc[key].values)
        plt.xlim(*(mean + 2 * std * np.array([-1, 1])))
        plt.xlabel(key)
        if i % n == 0:
            plt.ylabel("Difference in detection rate (%)")

plt.tight_layout()
plt.savefig(snakemake.output[0])
