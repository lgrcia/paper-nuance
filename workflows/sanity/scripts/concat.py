import pandas as pd
import yaml

results = []

for p, b, n in zip(
    snakemake.input.params,
    snakemake.input.bls,
    snakemake.input.nuance,
):
    params = yaml.full_load(open(p, "r"))
    bls = yaml.full_load(open(b, "r"))
    nuance = yaml.full_load(open(n, "r"))
    results.append(
        {
            "period": params["period"],
            "depth": params["depth"],
            "bls_period": bls["period"],
            "nuance_period": nuance["period"],
        }
    )

df = pd.DataFrame(results)
df.to_csv(snakemake.output[0], index=False)
