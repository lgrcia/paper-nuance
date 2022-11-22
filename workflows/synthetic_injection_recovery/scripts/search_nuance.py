import numpy as np
import yaml
import tinygp
from nuance import Nuance
from time import time as ctime

seed = int(snakemake.wildcards.seed)

time, flux, error = np.load(snakemake.input[0])
params = yaml.full_load(open(snakemake.input[1], "r"))
flux += 1

import tinygp
from nuance import Nuance

# the right one
kernel = tinygp.kernels.quasisep.SHO(params['omega'], params['quality'], sigma=params["sigma"])
gp = tinygp.GaussianProcess(kernel, time, diag=params['error']**2)

ct0 = ctime()

nu = Nuance(time, flux, gp)

t0s = time.copy()
Ds = np.linspace(0.01, 0.1, 8)
ll, z, vz = nu.linear_search(t0s, Ds, progress=False)

periods = np.linspace(1, 1.8, 1000)
llc, llv = nu.periodic_search(periods, progress=False)

i, j = np.unravel_index(np.argmax(llv), llv.shape)
p0 = periods[i]
t0, D = nu.best_periodic_transit(p0)

t = ctime() - ct0

result = {
    "t0": float(t0),
    "period": float(p0),
    "time": float(t)
}

yaml.safe_dump(result, open(snakemake.output[0], "w"))