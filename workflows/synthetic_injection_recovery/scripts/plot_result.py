import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(snakemake.input[0])

var = df.relative_duration
amp = df.relative_depth
detect = df.detected

plt.scatter(var, amp, c=detect, vmin=0., vmax=1.)
plt.savefig(snakemake.output[0])