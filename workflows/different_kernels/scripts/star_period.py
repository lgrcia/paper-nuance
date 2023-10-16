import numpy as np
from astropy.stats import LombScargle
import pickle as pkl
import yaml

data = pkl.load(open(snakemake.input[0], "rb"))


def rotation_period(time, flux):
    """rotation period based on LS periodogram"""

    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=1 / 5, maximum_frequency=1 / 0.1)
    period = 1 / frequency[np.argmax(power)]
    return period


period = rotation_period(data["time"], data["flux"])
yaml.safe_dump({"period": float(period)}, open(snakemake.output[0], "w"))
