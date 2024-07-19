# *nuance* paper 
Paper for [nuance](https://github.com/lgrcia/nuance), an algorithm to search for planetary transits in light curves featuring correlated noise, such as instrumental signals and stellar photometric variability ([pdf](./latex/paper.pdf))

> [!NOTE] 
> snakemake workflow is currently being cleaned up to allow full reproducibility.

## Running the workflows

Here are the different workflows that need to be ran:
- `workflows/plot_issues`
- `workflows/cleaning_snr`
- `workflows/principle` (only notebook for now)
- `workflows/control_test_bls`
- `workflows/synthetic-injection-recovery`
- `workflows/tess_injection_recovery`
- `workflows/comparison_toi`
- `workflows/benchmark`

Throughout these workflows, we try to protect the results as much as possible, as each takes couple hours/days to run. The `data` folders should be created and populated by the workflows. git versioned results are generated in the `results` folder, as well as in `figures`. Finally, if data files are necessary to start the workflow (such as target lists), they are placed in `static` folders.

Workflows are run in the same conda environment created with
```shell
conda env create -n paper-nuance -f environment.yml
```
where the local path to the nuance package might be adapted to your specific clone/pypi version

In this environment, each workflow is run with
```shell
snakemake -c12 --rerun-triggers input
```

## Moving all figures to a single folder

```zsh
sh ./copy_figures.sh
```
