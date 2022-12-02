# generate_params.py
import numpy as np
import yaml
import sys

argv = sys.argv

def make(seed=None, delta_v=None, tau_v=None):
    if seed is None:
        if delta_v is None:
            seed = np.random.randint(1, 1e5)
            np.random.seed(seed)
            delta_v = np.random.uniform(0.1, 10)
            tau_v = np.random.uniform(0.1, 5)
        
    params = {
        "t0" : 0.2,
        "duration" : 0.04,
        "depth" : 1e-2,
        "period" : 1.1,
        "error" : 1e-3,
    }

    if seed == -1:
        return make(tau_v=1000, delta_v=0.1)

    params.update({
        "relative_duration": float(tau_v),
        "relative_depth": float(delta_v),
        "omega" : float(np.pi/(params["duration"] * tau_v)),
        "quality" : float(np.random.uniform(10, 100)),
        "sigma" : float(params["depth"] * delta_v / 2),
        "seed" : int(seed),
    })
    
    return params

if __name__=="__main__":
    if len(argv) > 1:
        seed = int(argv[1])
    else:
        seed = None
        
    params = make(seed)
    yaml.safe_dump(params, open(f"data/params/{seed}.yaml", "w"))