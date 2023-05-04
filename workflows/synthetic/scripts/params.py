# generate_params.py
import sys

import numpy as np
import yaml

i = int(snakemake.wildcards.i)
n = int(snakemake.config["n"])
np.random.seed(i)
delta = np.random.uniform(0.1, 25)
tau = np.random.uniform(0.1, 10)

# delta = np.linspace(0.1, 25, n)
# tau = np.linspace(0.1, 10, n)
# delta, tau = np.meshgrid(delta, tau)
# delta = delta.flatten()[i]
# tau = tau.flatten()[i]

params = snakemake.config

params.update(
    {
        "tau": float(tau),
        "delta": float(delta),
        "omega": float(np.pi / (params["duration"] * tau)),
        "quality": float(np.random.uniform(10, 100)),
        "sigma": float(params["depth"] * delta / 2),
        "t0": 0.2,
    }
)

yaml.safe_dump(params, open(snakemake.output[0], "w"))
