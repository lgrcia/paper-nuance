import pickle

import numpy as np
from nuance.star import Star
from nuance.utils import periodic_transit

i = int(snakemake.wildcards.lc)
data_path, params_path = snakemake.input
data = pickle.load(open(data_path, "rb"))
periods, radii = pickle.load(open(params_path, "rb"))

star = Star(
    data["star_radius"], data["star_mass"], data["star_amplitude"], data["star_period"]
)

np.random.seed(i)
period = periods[i]
radius = radii[i]
tau = star.period2tau(period)
delta = star.radius2delta(radius)
duration = star.transit_duration(period)
depth = star.transit_depth(radius)

t0 = np.random.rand() * period

transit = depth * periodic_transit(data["time"], t0, duration, P=period)
injected_flux = data["flux"] + transit

data["flux"] = injected_flux
data["transit_t0"] = t0
data["transit_duration"] = duration
data["transit_depth"] = depth
data["delta"] = delta
data["tau"] = tau
data["planet_period"] = period
data["planet_radius"] = radius
output = snakemake.output[0]
pickle.dump(data, open(output, "wb"))
