from scipy.interpolate import BSpline, splrep
import numpy as np
import yaml

time, flux, error = np.load(snakemake.input.lc)
params = yaml.full_load(open(snakemake.input.params, "r"))


mask = np.ones_like(time, dtype=bool)


for i in range(2):
    tck = splrep(
        time[mask],
        flux[mask],
        w=1 / error[mask],
    )
    trend = BSpline(*tck)(time)
    mask &= np.abs(flux - trend) < 3 * np.std(flux - trend)

trend = BSpline(*tck)(time)

flatten_trend = BSpline(*tck)(time)
flatten_flux = flux - flatten_trend
flatten_flux -= np.mean(flatten_flux)
flatten_flux += 1.0

np.save(snakemake.output[0], [time, flatten_flux, error])