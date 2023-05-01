# I created this because i need some more info per target and don't want to modify the 
# download script, which would trigger a repprocessing of all the data

rule info_bis:
    input: "data/{target}/lc.fluxes"
    output: "figures/{target}/info.yaml"
    conda: "envs/base.yaml"
    priority: 9
    script: "scripts/plots/info.py"

rule plot_lc:
    input: "data/{target}/cleaned_lc.fluxes", "data/{target}/gp.yaml", "figures/{target}/info.yaml"
    output: "/Users/lgrcia/papers/phd-thesis/figures/nuance/{target}_cleaned_lc.pdf"
    conda: "envs/base.yaml"
    priority: 10
    script: "scripts/plots/plot_lc.py"

rule plot_comparison:
    input: "data/{target}/search_results.csv", "data/{target}/gp.yaml", "data/{target}/cleaned_lc.fluxes"
    output: "/Users/lgrcia/papers/phd-thesis/figures/nuance/{target}_search_comparison.pdf"
    conda: "envs/base.yaml"
    priority: 10
    script: "scripts/plots/plot_comparison.py"
