# generate_params.py
import numpy as np
import yaml
import sys

argv = sys.argv

def make(seed=None):
    if seed is None:
        seed = np.random.randint(1, 1e5)

    np.random.seed(seed)

    params = {
        "t0" : 0.2,
        "duration" : 0.04,
        "depth" : 1e-2,
        "period" : 1.1,
        "error" : 1e-3,
    }

    if seed == 0:
        params.update({
            "relative_duration": 1000,
            "relative_depth": 0.1,
        })

    else:
        params.update({
            "relative_duration": np.random.uniform(0.1, 20),
            "relative_depth": np.random.uniform(0.1, 20),
        })

    params.update({
        "omega" : 2*np.pi/(2*params["duration"] * params['relative_duration']),
        "quality" : np.random.uniform(10, 100),
        "sigma" : params["depth"] * (params['relative_depth']/2)**2,
        "seed" : seed,
    })

    yaml.dump(params, open(f"data/params/{params['seed']}.yaml", "w"))

if __name__=="__main__":
    if len(argv) > 1:
        seed = int(argv[1])
    else:
        seed = None

    make(seed)