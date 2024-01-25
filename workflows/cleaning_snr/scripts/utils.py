import numpy as np
from lightkurve import LightCurve

def snr(time, flux, t0, duration, error=None):
    _flux = flux - np.median(flux) + 1.0
    transit_model = (np.abs(time - t0) < duration / 2).astype(float) * -1
    X = np.vstack([np.ones_like(time), transit_model]).T
    w, _, _, _ = np.linalg.lstsq(X, _flux, rcond=None)
    exposure = np.median(np.diff(time))
    n = duration / exposure
    in_transit = np.abs(time - t0) < duration / 2
    cdpp = (
        LightCurve(time=time[~in_transit], flux=_flux[~in_transit])
        .estimate_cdpp(int(n), savgol_window=int(2 / exposure))
        .value
        * 1e-6
    )

    # sometimes CDPP is less than minimum it can be
    # this is a hack to make sure that doesn't happen
    if error is not None:
        min_cdpp = error / np.sqrt(n)
        cdpp = np.max([cdpp, min_cdpp])

    return w[-1]/cdpp