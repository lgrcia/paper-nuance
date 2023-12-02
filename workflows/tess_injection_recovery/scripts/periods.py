import numpy as np
import yaml
from nuance.utils import clean_periods

info = yaml.safe_load(open(snakemake.input[0], "r"))

periods = np.linspace(0.3, 6.0, 4000)
periods = clean_periods(periods, info["star_period"])

output = snakemake.output[0]
np.save(output, periods)
