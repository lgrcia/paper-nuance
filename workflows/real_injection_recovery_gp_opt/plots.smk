# I created this because i need some more info per target and don't want to modify the 
# download script, which would trigger a repprocessing of all the data

methods = ["bls_bspline", "bls_wotan3D", "bls_harmonics", "nuance"]

rule extra_info:
    output: "data/{target}/extra_info.yaml"
    priority: 10
    run:
        import yaml
        import pandas as pd
        # stellar parameters
        url = f"https://exofop.ipac.caltech.edu/tess/download_stellar.php?id={wildcards.target}"
        star = pd.read_csv(url, delimiter="|", index_col=1).iloc[0]
        yaml.dump(star.to_dict(),  open(output[0], "w"))

rule plot_lc:
    input: 
        fluxes="data/{target}/cleaned.fluxes",
        gp="data/{target}/gp.yaml",
        info="data/{target}/info.yaml"
    output: "test/cleaned/{target}.pdf"
    conda: "envs/base.yaml"
    priority: 10
    script: "scripts/plots/plot_lc.py"

rule plot_comparison:
    input: 
        results=[f"data/{{target}}/recovered/{method}/results.csv" for method in methods],
        info="data/{target}/info.yaml",
        first_injected="data/{target}/injected/0.fluxes",
        first_recovered=[f"data/{{target}}/recovered/{method}/0.params" for method in methods],
        gp="data/{target}/gp.yaml",
    params:
        methods = {
            "bls_bspline": r"{\ttfamily bspline + BLS}",
            "bls_wotan3D": r"{\ttfamily biweight + BLS}",
            "bls_harmonics": r"{\ttfamily harmonics + BLS}",
            "nuance": r"{\ttfamily nuance}",
        }
    output: "test/searched/{target}.pdf"
    params: 
        methods=methods
    conda: "envs/base.yaml"
    priority: 10
    script: "scripts/plot_comparison.py"


rule copy_figure:
    input: "test/{target}_search_comparison.pdf"
    output: "/Users/lgrcia/papers/phd-thesis/figures/nuance/{target}_search_comparison.pdf"
    shell: "cp {input} {output}"