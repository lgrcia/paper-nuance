import pandas as pd
import numpy as np
import yaml
import sys
import jax

sys.path.append("lib")
import utils
from nuance import Nuance

# Load data
params = yaml.full_load(open(snakemake.input[0], "r"))
df = pd.read_csv(snakemake.input[1])

time, flux, error = df.values.T
t_0 = time[0]
time -= t_0

X = utils.poly_X(time, 3)
kernel = utils.build_gp(params, time).kernel
nu = Nuance(time, flux, kernel=kernel, error=error, X=X)

# Mask flares
window = 30
r = flux - np.mean(flux)
mask = np.array(r < np.std(r) * 4)
ups = np.flatnonzero(~mask)
if len(ups) > 0:
    mask[
        np.hstack(
            [np.arange(max(u - window, 0), min(u + window, len(nu.time))) for u in ups]
        )
    ] = False

nu = Nuance(time[mask], flux[mask], error[mask], kernel=kernel, X=X[:, mask])

# linear search
t0s = time.copy()
Ds = np.linspace(0.01, 0.1, 10)
nu.linear_search(t0s, Ds)

# First search
periods = np.linspace(0.8, 12.0, 15000)
search = nu.periodic_search(periods)

# Second search
nu2 = nu.mask(*search.best)
search2 = nu2.periodic_search(periods)

yaml.safe_dump(
    [[float(p) for p in search.best], [float(p) for p in search2.best]],
    open(snakemake.output[0], "w"),
)
