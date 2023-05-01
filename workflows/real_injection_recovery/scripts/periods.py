import pickle

import numpy as np
from nuance.utils import clean_periods

lc_file = snakemake.input[0]
data = pickle.load(open(lc_file, "rb"))

periods = np.linspace(0.1, 11.0, 10000)
periods = clean_periods(periods, data["star_period"])

output = snakemake.output[0]
np.save(output, periods)
