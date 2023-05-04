import numpy as np

n_points = int(snakemake.wildcards.n_points)
n_X = int(snakemake.wildcards.n_X)

np.random.seed(n_points)
time = np.linspace(0, 3, n_points)
flux = np.random.normal(1.0, 0.01, size=len(time))
error = np.ones_like(time) * 0.0001

np.save(
    snakemake.output[0],
    np.array([time, flux, error]),
)
