from nuance import Nuance
from nuance.kernels import rotation
import yaml
import numpy as np

time, flux, error = np.load(snakemake.input[0])


def rotation_period(time, flux):
    """rotation period based on LS periodogram"""
    from astropy.timeseries import LombScargle

    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=1 / 5, maximum_frequency=1 / 0.1)
    period = 1 / frequency[np.argmax(power)]
    return period


star_period = rotation_period(time, flux)
nu = Nuance(time, flux, error.mean())

build_gp, init = rotation(star_period, error.mean(), long_scale=0.5)
optimize, mu, nll = nu.gp_optimization(build_gp)

gp_params = optimize(
    init, ["log_sigma", "log_short_scale", "log_short_sigma", "log_long_sigma"]
)
gp_params = optimize(gp_params)


nu_cleaned = nu.mask_flares(build_gp, gp_params, sigma=3.5)
yaml.safe_dump(
    {k: float(v) for k, v in gp_params.items()},
    open(snakemake.output[0], "w"),
)
np.save(snakemake.output[1], np.array([nu_cleaned.time, nu_cleaned.flux]))
