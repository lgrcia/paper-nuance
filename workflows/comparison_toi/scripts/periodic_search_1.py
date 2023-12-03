import pickle
import numpy as np
from nuance.combined import CombinedNuance
from time import time as timer
import yaml 

nu_s = CombinedNuance([pickle.load(open(f, "rb")) for f in snakemake.input])
len(nu_s.time)

t = timer()
combined_search = nu_s.periodic_search(snakemake.params[0])
t = timer() - t

np.save(snakemake.output[1], np.array(combined_search.best))
pickle.dump(combined_search, open(snakemake.output[0], "wb"))
yaml.safe_dump({"time": t}, open(snakemake.output[2], "w"))
