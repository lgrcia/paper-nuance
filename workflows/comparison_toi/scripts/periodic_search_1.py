import pickle
import numpy as np
from nuance.combined import CombinedNuance


nu_s = CombinedNuance([pickle.load(open(f, "rb")) for f in snakemake.input])
len(nu_s.time)

combined_search = nu_s.periodic_search(snakemake.params[0])
np.save(snakemake.output[1], np.array(combined_search.best))
pickle.dump(combined_search, open(snakemake.output[0], "wb"))
