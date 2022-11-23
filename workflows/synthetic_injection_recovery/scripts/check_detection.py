import numpy as np
import yaml

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

if __name__=="__main__":
    params = yaml.full_load(open(snakemake.input[0], "r"))
    result = yaml.full_load(open(snakemake.input[1], "r"))

    detected = right_candidate(
        params['t0'], 
        params['period'], 
        result['t0'], 
        result['period']
    )

    yaml.safe_dump({
        "detected": bool(detected),
        "relative_depth": params["relative_depth"],
        "relative_duration": params["relative_duration"],
        "true_t0": params['t0'], 
        "true_period": params['period'], 
        "detected_t0": result['t0'], 
        "detected_period": result['period']
    }, open(snakemake.output[0], "w"))