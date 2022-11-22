import pandas as pd
import yaml
import numpy as np

results = {
    "wotan_tls_detected": [],
    "nuance_detected": [],
    "relative_duration": [],
    "relative_depth": [],
}

def right_candidate(t0, period, true_t0, true_period, verbose=False):
    t0_check = np.abs((t0  - true_t0 + 0.5 * true_period) % true_period - 0.5 * true_period)%period < 0.01
    period_check = np.abs(period - true_period) < 0.1
    same = np.logical_and(t0_check, period_check)
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

for p, n, w in zip(snakemake.input.params, snakemake.input.nuance, snakemake.input.wotan):
    params = yaml.full_load(open(p, "r"))
    nuance_result = yaml.full_load(open(n, "r"))
    wotan_result = yaml.full_load(open(w, "r"))
    t0, period = params['t0'], params['period']
    results["wotan_tls_detected"].append(right_candidate( t0, period, wotan_result['t0'], wotan_result['period']))
    results["nuance_detected"].append(right_candidate( t0, period, nuance_result['t0'], nuance_result['period']))
    results["relative_depth"].append(params['relative_depth'])
    results["relative_duration"].append(params['relative_duration'])

df = pd.DataFrame(results)
df.to_csv(snakemake.output[0], index=False)