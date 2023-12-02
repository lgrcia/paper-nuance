import numpy as np
import yaml

time, flux, error = np.load(snakemake.input[0])
period_range = snakemake.config["period_range"]
duration = float(snakemake.config["duration"])
depths = yaml.full_load(open(snakemake.input[1]))

period = np.random.uniform(period_range[0], period_range[1])
depth = np.random.uniform(depths["min_depth"], depths["max_depth"])
t0 = np.random.uniform(0, period)


with open(snakemake.output[0], "w") as f:
    yaml.safe_dump({"period": float(period), "depth": float(depth), "t0": float(t0)}, f)
