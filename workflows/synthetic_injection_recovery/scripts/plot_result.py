import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(snakemake.input[0])

non_wotan = np.logical_and(df.nuance_detected, ~df["wotan_tls_detected"])
non_nuance = np.logical_and(~df.nuance_detected, df["wotan_tls_detected"])
print(f"V nuance - X wotan+tls: {np.count_nonzero(non_wotan)}")
print(f"X nuance - V wotan+tls: {np.count_nonzero(non_nuance)}")

var = df.relative_duration
amp = df.relative_depth
detect = df.nuance_detected
non_by_wotan = df[non_wotan][["relative_duration", "relative_depth"]].values.T
non_by_nuance = df[non_nuance][["relative_duration", "relative_depth"]].values.T

plt.scatter(var, amp, c=detect)
plt.scatter(*non_by_wotan, c="green")
plt.scatter(*non_by_nuance, c="red")
plt.savefig(snakemake.output[0])