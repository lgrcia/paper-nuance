# make_lightcurve.py
import jax
import numpy as np
import tinygp
import yaml
from nuance.utils import transit

jax.config.update("jax_enable_x64", True)
import sys

sys.path.append("./lib")
sys.path.append("../lib")
from utils import time


def make_lc(params, seed):
    kernel = tinygp.kernels.quasisep.SHO(
        params["omega"], params["quality"], sigma=params["sigma"]
    )
    gp = tinygp.GaussianProcess(kernel, time, diag=params["error"] ** 2)

    signal = transit(
        time, time.mean(), params["duration"], params["depth"], c=10, P=None
    )
    y = gp.sample(jax.random.PRNGKey(seed)) + signal

    error = np.ones_like(y) * params["error"]

    return time, y, error


if __name__ == "__main__":
    seed = int(snakemake.wildcards.seed)
    params = yaml.full_load(open(snakemake.input[0], "r"))
    time, y, error = make_lc(params, seed)
    np.save(snakemake.output[0], np.array([time, y, error]))
