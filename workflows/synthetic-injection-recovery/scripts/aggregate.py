import pickle

import pandas as pd
import yaml

results = []

for b, n, p in zip(
    snakemake.input.biweight_bls,
    snakemake.input.nuance,
    snakemake.input.params,
):
    params = yaml.full_load(open(p, "r"))
    bls = pickle.load(open(b, "rb"))
    nuance = pickle.load(open(n, "rb"))
    results.append(
        {
            "period": params["period"],
            "tau": params["tau"],
            "delta": params["delta"],
            "bls_period": bls["period"],
            "nuance_period": nuance["period"],
        }
    )

df = pd.DataFrame(results)
df.to_csv(snakemake.output[0], index=False)
