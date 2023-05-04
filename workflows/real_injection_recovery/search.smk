rule download:
    output: 
        fluxes = "data/{target}/original.fluxes",
        info = "data/{target}/info.yaml"
    priority: 0
    conda: "envs/base.yaml"
    script: "scripts/download.py"

rule optimize_gp:
    input: 
        fluxes = "data/{target}/original.fluxes",
        info = "data/{target}/info.yaml"
    output: "data/{target}/gp.yaml"
    priority: 1
    conda: "envs/base.yaml"
    script: "scripts/optimize_gp.py"

rule clean:
    input:
        fluxes = "data/{target}/original.fluxes",
        info = "data/{target}/info.yaml",
        gp = "data/{target}/gp.yaml"
    output: "data/{target}/cleaned.fluxes"
    priority: 2
    conda: "envs/base.yaml"
    script: "scripts/clean.py"

rule periods:
    input: "data/{target}/info.yaml"
    output: "data/{target}/periods.npy"
    priority: 3
    conda: "envs/base.yaml"
    script: "scripts/periods.py"

rule params:
    input:
        fluxes = "data/{target}/cleaned.fluxes",
        info = "data/{target}/info.yaml",
    output: "data/{target}/params.values"
    params:
        n=config["n"]
    priority: 3
    conda: "envs/base.yaml"
    script: "scripts/params.py"

rule inject:
    input:
        fluxes = "data/{target}/cleaned.fluxes",
        info = "data/{target}/info.yaml",
        params = "data/{target}/params.values"
    output: "data/{target}/injected/{lc}.fluxes"
    priority: 4
    conda: "envs/base.yaml"
    script: "scripts/inject.py"

rule tls_search:
    input:
        fluxes = "data/{target}/injected/{lc}.fluxes",
        info = "data/{target}/info.yaml",
        periods = "data/{target}/periods.npy"
    output:
        wotan3D = "data/{target}/recovered/wotan3D/{lc}.params",
        harmonics = "data/{target}/recovered/harmonics/{lc}.params",
    priority: 5
    conda: "envs/base.yaml"
    script: "scripts/tls_search.py"

rule bspline_tls:
    input:
        fluxes = "data/{target}/injected/{lc}.fluxes",
        info = "data/{target}/info.yaml",
        periods = "data/{target}/periods.npy"
    output: "data/{target}/recovered/bspline/{lc}.params"
    priority: 5
    conda: "envs/base.yaml"
    script: "scripts/bspline_tls.py"

rule bls:
    input:
        fluxes = "data/{target}/injected/{lc}.fluxes",
        info = "data/{target}/info.yaml",
        periods = "data/{target}/periods.npy"
    output:
        wotan3D = "data/{target}/recovered/bls_wotan3D/{lc}.params",
        harmonics = "data/{target}/recovered/bls_harmonics/{lc}.params",
        bspline = "data/{target}/recovered/bls_bspline/{lc}.params",
    priority: 5
    conda: "envs/base.yaml"
    script: "scripts/bls.py"

rule nuance_search:
    input: 
        fluxes = "data/{target}/injected/{lc}.fluxes",
        info = "data/{target}/info.yaml",
        periods = "data/{target}/periods.npy",
        gp = "data/{target}/gp.yaml"
    output: "data/{target}/recovered/nuance/{lc}.params"
    priority: 6
    conda: "envs/base.yaml"
    script: "scripts/nuance_search.py"

rule concatenate:
    input: 
        recovered = [f"data/{{target}}/recovered/{{tool}}/{lc}.params" for lc in idxs],
        injected = [f"data/{{target}}/injected/{lc}.fluxes" for lc in idxs],
    output: "data/{target}/recovered/{tool}/results.csv"
    priority: 7
    conda: "envs/base.yaml"
    script: "scripts/concatenate.py"
