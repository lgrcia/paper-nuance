import pickle
import numpy as np
from nuance.combined import CombinedNuance


nu_s = CombinedNuance([pickle.load(open(f, "rb")) for f in snakemake.input.nus])
best = np.load(snakemake.input.best)
nu_s2 = nu_s.mask_transit(*best)

combined_search = nu_s2.periodic_search(snakemake.params[0])
pickle.dump(combined_search, open(snakemake.output[0], "wb"))
