import jax
import numpy as np
import tinygp
import yaml
from nuance import Nuance

jax.config.update("jax_enable_x64", True)

time, flux, error = np.load(snakemake.input.lc)
params = yaml.full_load(open(snakemake.input.params, "r"))

periods = np.linspace(0.8, 3, 1000)

kernel = tinygp.kernels.quasisep.SHO(
    params["omega"], params["quality"], sigma=params["sigma"]
)
gp = tinygp.GaussianProcess(kernel, time, diag=params["error"] ** 2)
nu = Nuance(time, flux, gp=gp)
duration = snakemake.config["duration"]
Ds = np.array([duration - 0.001, duration])
nu.linear_search(time.copy(), Ds)
search = nu.periodic_search(periods)

t0, _, period = search.best

pickle.dump(
    {
        "t0": t0,
        "period": float(period),
        "power": search.Q_snr,
    },
    open(snakemake.output[0], "wb"),
)
