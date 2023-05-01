import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

data_path = snakemake.input[0]
gp_path = snakemake.input[1]
info_path = snakemake.input[2]

data = pickle.load(open(data_path, "rb"))
gp_params = yaml.full_load(open(gp_path, "r"))
info = yaml.full_load(open(info_path, "r"))

build_gp, init = rotation(data["star_period"], np.mean(data["error"]), long_scale=0.5)
gp = build_gp(gp_params, data["time"])

nu = Nuance(data["time"], data["flux"], gp=gp)
optimize, mu, nll = nu.gp_optimization(build_gp)


plt.figure(None, (8.5, 3))
plt.xlabel(
    f"time - {info['first_exposure_time']:.2f} ${info['first_exposure_time_format']}$"
)
plt.ylabel("differential flux")
plt.title(f"TIC {info['tic']} PDCSAP light curve (sector {int(info['sector'])})")
plt.plot(data["time"], data["flux"], ".", c="0.8", label="cleaned data")
gp_mean = mu(gp_params)
idxs = [0, *np.flatnonzero(np.diff(data["time"]) > 10 / 60 / 24)]
for i in range(len(idxs) - 1):
    x = data["time"][idxs[i] + 1 : idxs[i + 1]]
    y = gp_mean[idxs[i] + 1 : idxs[i + 1]]
    plt.plot(x, y, "k", label="GP mean" if i == 0 else None)
plt.legend(loc="lower right")
plt.xlim(0, 3)

plt.tight_layout()
plt.savefig(snakemake.output[0])
