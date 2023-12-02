import yaml
import numpy as np

time, flux, error = np.load(snakemake.input[0])
period_range = snakemake.config["period_range"]
snr_range = [4.0, 30.0]
duration = float(snakemake.config["duration"])

# calculate the minimum and maximum depth
minimum_number_of_transit_points = len(time) * duration / period_range[1]
min_depth = snr_range[0] * np.median(error) / np.sqrt(minimum_number_of_transit_points)

maximum_number_of_transit_points = len(time) * duration / period_range[0]
max_depth = snr_range[1] * np.median(error) / np.sqrt(maximum_number_of_transit_points)

# save the results
with open(snakemake.output[0], "w") as f:
    yaml.safe_dump({"min_depth": float(min_depth), "max_depth": float(max_depth)}, f)
