import pickle

import numpy as np
import pandas as pd
import yaml


def right_candidate(t0, period, true_t0, true_period, verbose=False):
    t0_check = (
        np.abs((t0 - true_t0 + 0.5 * true_period) % true_period - 0.5 * true_period)
        % period
        < 0.01
    )
    period_check = np.abs(period - true_period) < 0.01
    period_check |= np.abs(2 * period - true_period) < 0.01
    period_check |= np.abs(period / 2 - true_period) < 0.01
    same = period_check  # np.logical_and(t0_check, period_check)
    if verbose:
        if not same:
            if not t0_check:
                output = f"t0 differ: {t0:.2e} {true_t0:.2e}"
            if not period_check:
                output = f"period differ: {period:.2e} {true_period:.2e}"
        else:
            output = "match"
        return same, output
    else:
        return same


def get_result(injected_file, recovered_file, fresh_file):
    injected = pickle.load(open(injected_file, "rb"))
    recovered = pickle.load(open(recovered_file, "rb"))
    fresh = yaml.safe_load(open(fresh_file, "r"))
    t0, period, snr = recovered["t0"], recovered["period"], recovered["snr"]
    true_t0, true_period = injected["transit_t0"], injected["planet_period"]
    return {
        "t0": t0,
        "period": period,
        "true_t0": true_t0,
        "true_period": true_period,
        "found": right_candidate(t0, period, true_t0, true_period),
        "radius": injected["planet_radius"],
        "tau": injected["tau"],
        "delta": injected["delta"],
        "fresh_snr": fresh["snr"],
        "snr": snr,
    }


recovered = snakemake.input.recovered
injected = snakemake.input.injected
fresh = snakemake.input.fresh

results = []

for i, (a, b) in enumerate(zip(injected, recovered)):
    results.append(get_result(a, b, fresh))
results = pd.DataFrame(results)
results.to_csv(snakemake.output[0], index=False)
