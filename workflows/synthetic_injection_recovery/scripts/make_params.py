# generate_params.py
import numpy as np
import yaml
import sys

argv = sys.argv

def make(seed=None, amp=None, var=None):
    if seed is None:
        if amp is None:
            seed = np.random.randint(1, 1e5)
            np.random.seed(seed)
            amp = np.random.uniform(0.1, 10)
            var = np.random.uniform(0.1, 5)
        
    params = {
        "t0" : 0.2,
        "duration" : 0.04,
        "depth" : 1e-2,
        "period" : 1.1,
        "error" : 1e-3,
    }

    if seed == -1:
        return make(var=1000, amp=0.1)

    params.update({
        "relative_duration": float(var),
        "relative_depth": float(amp),
        "omega" : float(2*np.pi/(2*params["duration"] * var)),
        "quality" : float(np.random.uniform(10, 100)),
        "sigma" : float(params["depth"] * (amp/2)**2),
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