import numpy as np
import yaml

time, flux, error = np.load(snakemake.input[0])
period_range = snakemake.config["period_range"]
snr_range = [4.0, 30.0]
duration = float(snakemake.config["duration"])
n = int(snakemake.config["n"])

# calculate the minimum and maximum depth
minimum_number_of_transit_points = len(time) * duration / period_range[1]
min_depth = snr_range[0] * np.median(error) / np.sqrt(minimum_number_of_transit_points)

maximum_number_of_transit_points = len(time) * duration / period_range[0]
max_depth = snr_range[1] * np.median(error) / np.sqrt(maximum_number_of_transit_points)


periods = np.linspace(period_range[0], period_range[1], n)
depths = np.linspace(min_depth, max_depth, n)

# grid
periods, depths = np.meshgrid(periods, depths)
# as pairs
periods = periods.flatten()
depths = depths.flatten()
period_depth = np.array([periods, depths]).T

np.save(snakemake.output[0], period_depth)
