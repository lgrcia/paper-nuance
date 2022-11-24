import pandas as pd
import yaml
import numpy as np

results = {
    "snr": [],
    "relative_duration": [],
    "relative_depth": [],
}

for r in snakemake.input:
    result = yaml.full_load(open(r, "r"))
    results["relative_depth"].append(result['relative_depth'])
    results["relative_duration"].append(result['relative_duration'])
    results["snr"].append(result["snr"])

df = pd.DataFrame(results)
df.to_csv(snakemake.output[0], index=False)