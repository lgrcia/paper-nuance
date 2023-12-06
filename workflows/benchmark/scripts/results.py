import matplotlib.pyplot as plt
import numpy as np
import yaml

nuance = np.array([yaml.safe_load(open(f)) for f in snakemake.input.nuance])
bls = np.array([yaml.safe_load(open(f))for f in snakemake.input.bls])

result = {
    "points": np.array(snakemake.params.points).tolist(),
    "nuance_linear": [n["linear"] for n in nuance],
    "nuance_periodic": [b["biweight"] for b in bls],
    "biweight": [n["all"] - n["linear"] for n in nuance],
    "bls": [b["bls"] - b["biweight"] for b in bls]
}

with open(snakemake.output[0], "w") as f:
    yaml.dump(result, f)