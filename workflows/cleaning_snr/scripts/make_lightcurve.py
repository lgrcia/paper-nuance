# make_lightcurve.py
import jax
import numpy as np
import tinygp
import yaml
from nuance.utils import transit

jax.config.update("jax_enable_x64", True)

def build_gp(time, params):
    kernel = tinygp.kernels.quasisep.SHO(
        params["omega"], params["quality"], sigma=params["sigma"]
    )
    gp = tinygp.GaussianProcess(kernel, time, diag=params["error"] ** 2)
    return gp

def make_lc(time, params, seed):
    gp = build_gp(time, params)

    signal = transit(
        time, time.mean(), params["duration"], params["depth"], c=50000, P=None
    )
    y = gp.sample(jax.random.PRNGKey(seed)) + signal

    error = np.ones_like(y) * params["error"]

    return time, y, error


if __name__ == "__main__":
    config = snakemake.config
    time = np.arange(0, config["length"], config["exposure"])
    seed = int(snakemake.wildcards.seed)
    params = yaml.full_load(open(snakemake.input[0], "r"))
    time, y, error = make_lc(time, params, seed)
    np.save(snakemake.output[0], np.array([time, y, error]))
