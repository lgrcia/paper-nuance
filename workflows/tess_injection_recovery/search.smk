rule download:
    output:
        fluxes="data/{target}/original.fluxes",
        info="data/{target}/info.yaml",
    script:
        "scripts/download.py"


rule optimize_gp_to_clean:
    input:
        fluxes="data/{target}/original.fluxes",
        info="data/{target}/info.yaml",
    output:
        "data/{target}/gp.yaml",
    script:
        "scripts/optimize_gp.py"


rule clean:
    input:
        fluxes="data/{target}/original.fluxes",
        info="data/{target}/info.yaml",
        gp="data/{target}/gp.yaml",
    output:
        "data/{target}/cleaned.fluxes",
    script:
        "scripts/clean.py"


rule periods:
    input:
        "data/{target}/info.yaml",
    output:
        "data/{target}/periods.npy",
    script:
        "scripts/periods.py"


rule params:
    input:
        fluxes="data/{target}/cleaned.fluxes",
        info="data/{target}/info.yaml",
    output:
        "data/{target}/params.values",
    params:
        n=config["n"],
    script:
        "scripts/params.py"


rule inject:
    input:
        fluxes="data/{target}/cleaned.fluxes",
        info="data/{target}/info.yaml",
        params="data/{target}/params.values",
    output:
        "data/{target}/injected/{lc}.fluxes",
    script:
        "scripts/inject.py"


rule optimize_gp:
    input:
        fluxes="data/{target}/injected/{lc}.fluxes",
        info="data/{target}/info.yaml",
    output:
        "data/{target}/gp/{lc}.yaml",
    script:
        "scripts/optimize_gp.py"


rule bls:
    input:
        fluxes="data/{target}/injected/{lc}.fluxes",
        info="data/{target}/info.yaml",
        periods="data/{target}/periods.npy",
        gp="data/{target}/gp.yaml",
    output:
        wotan3D="data/{target}/recovered/bls_wotan3D/{lc}.params",
        harmonics="data/{target}/recovered/bls_harmonics/{lc}.params",
        bspline="data/{target}/recovered/bls_bspline/{lc}.params",
        bens="data/{target}/recovered/bens/{lc}.params",
        gp="data/{target}/recovered/gp/{lc}.params",
    script:
        "scripts/bls.py"

rule nuance_search:
    input:
        fluxes="data/{target}/injected/{lc}.fluxes",
        info="data/{target}/info.yaml",
        periods="data/{target}/periods.npy",
        gp="data/{target}/gp/{lc}.yaml",
    output:
        "data/{target}/recovered/nuance/{lc}.params",
    script:
        "scripts/nuance_search.py"


rule concatenate:
    input:
        recovered=[f"data/{{target}}/recovered/{{method}}/{lc}.params" for lc in idxs],
        injected=[f"data/{{target}}/injected/{lc}.fluxes" for lc in idxs],
        fresh = "data/{target}/fresh_search.yaml",
    output:
        "data/{target}/recovered/{method}/results.csv",
    script:
        "scripts/concatenate.py"

rule fresh_search:
    input:
        fluxes="data/{target}/cleaned.fluxes",
        info="data/{target}/info.yaml",
        periods="data/{target}/periods.npy",
        gp="data/{target}/gp.yaml",
    output: 
        "figures/check/{target}.pdf",
        "data/{target}/fresh_search.yaml"
    script:
        "scripts/fresh_search.py"
       
