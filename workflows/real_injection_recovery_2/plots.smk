# I created this because i need some more info per target and don't want to modify the 
# download script, which would trigger a repprocessing of all the data

rule plot_lc:
    input: 
        fluxes="data/{target}/cleaned.fluxes",
        gp="data/{target}/gp.yaml",
        info="data/{target}/info.yaml"
    output: "figures/{target}_cleaned_fluxes.pdf"
    conda: "envs/base.yaml"
    priority: 10
    script: "scripts/plots/plot_lc.py"

rule plot_comparison:
    input: "data/{target}/search_results.csv", "data/{target}/gp.yaml", "data/{target}/cleaned_lc.fluxes"
    output: "/Users/lgrcia/papers/phd-thesis/figures/nuance/{target}_search_comparison.pdf"
    conda: "envs/base.yaml"
    priority: 10
    script: "scripts/plots/plot_comparison.py"
