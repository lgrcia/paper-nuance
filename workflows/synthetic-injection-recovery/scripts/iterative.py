import numpy as np
import yaml

time, flux, error = np.load(snakemake.input.lc)
params = yaml.full_load(open(snakemake.input.params, "r"))

def bens_detrend(time, flux, n=5):
    def rotation_period(time, flux):
        """rotation period based on LS periodogram"""
        from astropy.stats import LombScargle

        ls = LombScargle(time, flux)
        frequency, power = ls.autopower(
            minimum_frequency=1 / 5, maximum_frequency=1 / 0.1
        )
        period = 1 / frequency[np.argmax(power)]
        return period

    def subtract_sinusoid(time, flux, period):
        X = np.vstack(
            [
                np.ones(len(time)),
                np.cos(2 * np.pi / period * time),
                np.sin(2 * np.pi / period * time),
            ]
        ).T

        w = np.linalg.lstsq(X, flux, rcond=None)[0]
        model = np.dot(X, w)

        return flux - model, model

    _flux = flux.copy()
    model = np.zeros_like(flux)

    for _ in range(n):
        period = rotation_period(time, _flux)
        _flux, _model = subtract_sinusoid(time, _flux, period)
        model += _model

    return _flux + 1.0, model


flatten_flux, flatten_trend = bens_detrend(time, flux, n=8)

np.save(snakemake.output[0], [time, flatten_flux, error])