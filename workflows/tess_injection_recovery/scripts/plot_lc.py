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
raw_path = snakemake.input.raw

original_data = pickle.load(open(raw_path, "rb"))
data = pickle.load(open(data_path, "rb"))
gp_params = yaml.full_load(open(gp_path, "r"))
info = yaml.full_load(open(info_path, "r"))

build_gp, init = rotation(info["star_period"], np.mean(data["error"]), long_scale=0.5)
gp = build_gp(gp_params, data["time"])

nu = Nuance(data["time"], data["flux"], gp=gp)
optimize, mu, nll = nu.gp_optimization(build_gp)
gp_mean = mu(gp_params)


plt.figure(None, (8.5, 5))
ax = plt.subplot(211)
# ax.xaxis.set_ticklabels([])
ax.set_title(
    f"TIC {info['tic']} PDCSAP cleaned light curve (sector {int(info['sector'])})"
)
plt.plot(
    original_data["time"],
    original_data["flux"],
    ".",
    c="k",
    label="original data",
    ms=3,
)
plt.plot(data["time"], data["flux"], ".", c="0.8", ms=3.5)
plt.ylabel("diff. flux")
plt.xlim(original_data["time"][0], original_data["time"][-1])
ylim = ax.get_ylim()
plt.legend()

plt.subplot(212)
plt.plot(
    data["time"], data["flux"], ".", c="0.8", label="trimmed and cleaned data", ms=3.5
)
split_idxs = [
    0,
    *np.flatnonzero(np.diff(data["time"]) > 10 / 60 / 24),
    len(data["time"]),
]

_ = True
for i in range(len(split_idxs) - 1):
    x = data["time"][split_idxs[i] + 1 : split_idxs[i + 1]]
    y = gp_mean[split_idxs[i] + 1 : split_idxs[i + 1]]
    plt.plot(x, y, "k", label="GP mean" if _ else None)
    _ = False
plt.xlim(data["time"][0], data["time"][-1])
plt.legend()
plt.ylabel("diff. flux")
minmax = np.max(np.abs(1 - data["flux"])) * 1.3
plt.ylim(1 - minmax, 1 + minmax)
plt.xlabel(f"time - {info['time0']:.2f} ${info['time0_format']}$")
plt.xlim(data["time"][0], data["time"][-1])

plt.tight_layout()
plt.savefig(snakemake.output[0])
