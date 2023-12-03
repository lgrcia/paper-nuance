import numpy as np
from nuance import Nuance
from nuance.kernels import rotation
import yaml
import pickle
from time import time as timer

durations = snakemake.params[0]
nus = []

time, flux = np.load(snakemake.input[0])
gp_params = yaml.safe_load(open(snakemake.input[1]))
build_gp, init = rotation(1, long_scale=0.5)
gp = build_gp(gp_params, time)

t = timer()
nu = Nuance(time, flux, gp=gp)
nu.linear_search(time.copy(), durations)
t = timer() - t

del nu.eval_m
pickle.dump(nu, open(snakemake.output[0], "wb"))
yaml.safe_dump({"time": t}, open(snakemake.output[1], "w"))
