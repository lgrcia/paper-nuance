import yaml
import numpy as np
import os
from multiprocessing import cpu_count

cores = int(eval(snakemake.wildcards.cores))
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"

import jax

from astropy.timeseries import BoxLeastSquares
from nuance import Nuance
from time import time as _time



n_durations = int(eval(snakemake.wildcards.durations))
periods = np.linspace(*[eval(i) for i in snakemake.wildcards.periods.split("_")])
durations = np.linspace(0.01, 0.05, n_durations)
time, flux, error = np.load(snakemake.input[0])
n_points = int(snakemake.wildcards.n_points)

times = {}

nu = Nuance(time, flux, error)
t0 = _time()
nu.linear_search(time, durations)
times["linear"] = float(_time() - t0)
search = nu.periodic_search(periods)
times["all"] = float(_time() - t0)
times["cpu_counts"] = jax.device_count(), cores
yaml.safe_dump(times, open(snakemake.output[0], "w"))
