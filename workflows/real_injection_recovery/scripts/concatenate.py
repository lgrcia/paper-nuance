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


def get_result(f, tool):
    d = yaml.safe_load(open(f, "r"))
    return {
        f"{tool}_found_t0": d["found_t0"],
        f"{tool}_found_period": d["found_period"],
        f"{tool}_found": right_candidate(
            d["found_t0"], d["found_period"], d["transit_t0"], d["planet_period"]
        ),
    }, d


nuance_files = snakemake.input.nuance
tls_files = snakemake.input.tls

results = []

for i, (nuance_file, tls_file) in enumerate(zip(nuance_files, tls_files)):
    nuance_results, injected = get_result(nuance_file, "nuance")
    tls_results, _ = get_result(tls_file, "tls")
    results.append({**nuance_results, **tls_results, **injected})

results = pd.DataFrame(results)

results.to_csv(snakemake.output[0], index=False)
