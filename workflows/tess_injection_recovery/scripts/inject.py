import pickle

import numpy as np
import yaml
from nuance.star import Star
from nuance.utils import periodic_transit

i = int(snakemake.wildcards.lc)
data = pickle.load(open(snakemake.input.fluxes, "rb"))
periods, radii = pickle.load(open(snakemake.input.params, "rb"))
info = yaml.safe_load(open(snakemake.input.info, "r"))

star = Star(
    info["star_radius"], info["star_mass"], info["star_amplitude"], info["star_period"]
)

np.random.seed(i)
period = periods[i]
radius = radii[i]
tau = star.period2tau(period)
delta = star.radius2delta(radius)
duration = star.transit_duration(period)
depth = star.transit_depth(radius)

t0 = np.random.rand() * period

transit = depth * periodic_transit(data["time"], t0, duration, P=period, c=500)
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
