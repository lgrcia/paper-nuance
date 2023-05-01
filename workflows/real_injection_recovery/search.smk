rule download:
    output: "data/{target}/lc.fluxes"
    priority: 0
    conda: "envs/base.yaml"
    script: "scripts/download.py"

rule optimize_gp:
    input: "data/{target}/lc.fluxes"
    output: "data/{target}/gp.yaml"
    priority: 1
    conda: "envs/base.yaml"
    script: "scripts/optimize_gp.py"

rule clean_flares:
    input: "data/{target}/lc.fluxes", "data/{target}/gp.yaml"
    output: "data/{target}/cleaned_lc.fluxes"
    priority: 2
    conda: "envs/base.yaml"
    script: "scripts/clean_flares.py"

rule periods:
    input: "data/{target}/cleaned_lc.fluxes"
    output: "data/{target}/periods.npy"
    priority: 3
    conda: "envs/base.yaml"
    script: "scripts/periods.py"

rule params:
    input: "data/{target}/cleaned_lc.fluxes"
    output: "data/{target}/params.values"
    params:
        n=config["n"]
    priority: 3
    conda: "envs/base.yaml"
    script: "scripts/params.py"

rule inject:
    input: "data/{target}/cleaned_lc.fluxes", "data/{target}/params.values"
    output: "data/{target}/injected/{lc}.fluxes"
    priority: 4
    conda: "envs/base.yaml"
    script: "scripts/inject.py"

rule tls_search:
    input: "data/{target}/injected/{lc}.fluxes", "data/{target}/periods.npy"
    output: "data/{target}/tls_search/{lc}.yaml"
    priority: 5
    conda: "envs/base.yaml"
    script: "scripts/tls_search.py"

rule nuance_search:
    input: "data/{target}/injected/{lc}.fluxes", "data/{target}/gp.yaml", "data/{target}/periods.npy"
    output: "data/{target}/nuance_search/{lc}.yaml"
    priority: 6
    conda: "envs/base.yaml"
    script: "scripts/nuance_search.py"

rule concatenate:
    input: 
        nuance = [f"data/{{target}}/nuance_search/{lc}.yaml" for lc in idxs],
        tls = [f"data/{{target}}/tls_search/{lc}.yaml" for lc in idxs]
    output: "data/{target}/search_results.csv"
    priority: 7
    conda: "envs/base.yaml"
    script: "scripts/concatenate.py"
