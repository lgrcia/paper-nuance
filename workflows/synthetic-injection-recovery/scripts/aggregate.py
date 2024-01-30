import pickle

import pandas as pd
import yaml

methods = ["nuance", *snakemake.params["methods"]]
results = {
    method: [pickle.load(open(f, "rb"))["period"] for f in snakemake.input[method]] for method in methods
}
results["tau"] = [yaml.full_load(open(p, "r"))["tau"] for p in snakemake.input.params]
results["delta"] = [yaml.full_load(open(p, "r"))["delta"] for p in snakemake.input.params]
results["period"] = [yaml.full_load(open(p, "r"))["period"] for p in snakemake.input.params]

df = pd.DataFrame(results)
df.to_csv(snakemake.output[0], index=False)
