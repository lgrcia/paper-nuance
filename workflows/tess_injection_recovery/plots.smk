# I created this because i need some more info per target and don't want to modify the 
# download script, which would trigger a repprocessing of all the data

rule extra_info:
    output: "data/{target}/extra_info.yaml"
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
        info="data/{target}/info.yaml",
        raw="data/{target}/original.fluxes"
    output: "_figures/cleaned/{target}.pdf"
    script: "scripts/plot_lc.py"

rule plot_single_comparison:
    input: 
        results=[f"data/{{target}}/recovered/{method}/results.csv" for method in methods.keys()],
        info="data/{target}/info.yaml",
        first_injected="data/{target}/injected/0.fluxes",
        first_recovered=[f"data/{{target}}/recovered/{method}/0.params" for method in methods.keys()],
        gp="data/{target}/gp.yaml",
    params:
        methods = methods
    output: "_figures/searched/{target}.pdf"
    script: "scripts/plot_comparison.py"

# rule plot_results:
#     input: 
#         results=[f"data/{{target}}/recovered/{method}/results.csv" for method in methods.keys()],
#     params:
#         methods = methods
#     output: "figures/searched/{target}.pdf"
#     script: "scripts/plot_results.py"
