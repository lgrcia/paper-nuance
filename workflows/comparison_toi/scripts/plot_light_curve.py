import matplotlib.pyplot as plt
import numpy as np
import yaml
from nuance import Nuance
from nuance.kernels import rotation

time, flux, _ = np.load(snakemake.input[0])
cleaned_time, cleaned_flux = np.load(snakemake.input[1])
gp_params = yaml.safe_load(open(snakemake.input[2], "r"))

build_gp, init = rotation(0.5, None, long_scale=0.5)
gp = build_gp(gp_params, cleaned_time)
nu = Nuance(cleaned_time, cleaned_flux, gp=gp)
_, mu, _ = nu.gp_optimization(build_gp)

plt.figure(figsize=(9, 3))
plt.plot(time, flux, ".", c="k", ms=2, label="masked flares")
plt.plot(cleaned_time, cleaned_flux, ".", c="0.85", ms=3, label="cleaned light curve")

gp_mean = mu(gp_params)
split_idxs = [
    0,
    *np.flatnonzero(np.diff(cleaned_time) > 0.5),
    len(time),
]

_ = True
for i in range(len(split_idxs) - 1):
    x = cleaned_time[split_idxs[i] + 1 : split_idxs[i + 1]]
    y = gp_mean[split_idxs[i] + 1 : split_idxs[i + 1]]
    plt.plot(x, y, "k", label="nuance GP mean" if _ else None, lw=1)
    _ = False

plt.xlim(time.min(), time.max())
plt.ylabel("Normalized flux")
plt.xlabel("Time ($BTJD_{TDB}$)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(snakemake.output[0])
