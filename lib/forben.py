import numpy as np
from astropy.stats import LombScargle


def bens_detrend(time, flux, n=5):
    """iterative simple harmonics detrending"""

    def rotation_period(time, flux):
        """rotation period based on LS periodogram"""

        ls = LombScargle(time, flux)
        frequency, power = ls.autopower(
            minimum_frequency=1 / 10, maximum_frequency=1 / 0.1
        )
        period = 1 / frequency[np.argmax(power)]
        return period

    def subtract_sinusoid(time, flux, period):
        """subtract a sinusoid with period from flux"""
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
