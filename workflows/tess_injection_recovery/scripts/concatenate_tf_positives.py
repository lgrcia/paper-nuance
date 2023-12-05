import numpy as np

results = snakemake.input
result = np.swapaxes(np.stack([np.load(r) for r in results], axis=-1), 1, 2)

np.save(snakemake.output[0], result)

