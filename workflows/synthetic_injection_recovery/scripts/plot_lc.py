import matplotlib.pyplot as plt
import numpy as np

time, flux, _ = np.load(snakemake.input[0])
plt.figure(None, (15, 5))
plt.plot(time, flux)
plt.savefig(snakemake.output[0])