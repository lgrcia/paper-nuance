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
if len(time) > 0:
    t_0 = time[0]
    time -= t_0

    X = utils.poly_X(time, 3)
    kernel = utils.build_gp(params, time).kernel
    nu = Nuance(time, flux, kernel=kernel, error=error, X=X)

    # Mask flares
    # In what Fran Pozuelos sent me, the falres are sigma clipped. To remove their
    # full extended signal, I locate and mask 30 points on each side of "holes" in
    # time
    window = 30
    mask = np.pad(np.array(np.diff(time) < 3 * np.median(np.diff(time))), (0, 1))
    ups = np.flatnonzero(~mask)
    if len(ups) > 0:
        mask[
            np.hstack(
                [
                    np.arange(max(u - window, 0), min(u + window, len(nu.time)))
                    for u in ups
                ]
            )
        ] = False

    nu = Nuance(time[mask], flux[mask], error[mask], kernel=kernel, X=X[:, mask])

    # linear search
    t0s = time.copy()
    Ds = np.linspace(0.01, 0.07, 6)
    nu.linear_search(t0s, Ds)

    # We mask directly the first planet signal
    nu = nu.mask(0.9236153776066658, 0.02, 1.2390692712847524)

    # Search of the second
    periods = np.linspace(0.9, 10.1, 8000)
    search = nu.periodic_search(periods)

    yaml.safe_dump([float(p) for p in search.best], open(snakemake.output[0], "w"))
else:
    yaml.safe_dump([0], open(snakemake.output[0], "w"))
