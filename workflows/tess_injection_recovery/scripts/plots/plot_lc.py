import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

data_path = snakemake.input.fluxes
gp_path = snakemake.input.gp
info_path = snakemake.input.info

data = pickle.load(open(data_path, "rb"))
gp_params = yaml.full_load(open(gp_path, "r"))
info = yaml.full_load(open(info_path, "r"))

build_gp, init = rotation(info["star_period"], np.mean(data["error"]), long_scale=0.5)
gp = build_gp(gp_params, data["time"])

nu = Nuance(data["time"], data["flux"], gp=gp)
optimize, mu, nll = nu.gp_optimization(build_gp)


plt.figure(None, (8.5, 3))
plt.xlabel(f"time - {info['time0']:.2f} ${info['time0_format']}$")
plt.ylabel("differential flux")
plt.title(f"TIC {info['tic']} PDCSAP light curve (sector {int(info['sector'])})")
plt.plot(data["time"], data["flux"], ".", c="0.8", label="cleaned data")
gp_mean = mu(gp_params)
idxs = [0, *np.flatnonzero(np.diff(data["time"]) > 10 / 60 / 24), len(data["time"])]
for i in range(len(idxs) - 1):
    x = data["time"][idxs[i] + 1 : idxs[i + 1]]
    y = gp_mean[idxs[i] + 1 : idxs[i + 1]]
    plt.plot(x, y, "k", label="GP mean" if i == 0 else None)
plt.legend(loc="lower right")
plt.xlim(data["time"][0], data["time"][0] + 3)
std = np.std(data["flux"]) * 5
plt.ylim(1 - std, 1 + std)

plt.tight_layout()
plt.savefig(snakemake.output[0])
