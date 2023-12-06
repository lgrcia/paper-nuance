import numpy as np
import yaml

tps = {}
fps = {}
detectables = {}
stats = {}

for i, f in enumerate(snakemake.input):
    m = f.split("/")[-1].split(".")[0]
    tau, delta, tp, fp, detectable = np.load(f)
    tps[m] = tp
    fps[m] = fp
    detectables[m] = detectable.astype(bool)

main_method = "nuance"
methods = tps.keys()
other_methods = [method for method in methods if method != main_method]

others_true_positives = np.max([tps[method] for method in other_methods], 0)
others_false_positives = np.min([fps[method] for method in other_methods], 0)

mask = tau < 5
nuance_better_true_positives = tps[main_method][mask] >= others_true_positives[mask]
nuance_better_false_positives = fps[main_method][mask] <= others_false_positives[mask]
nuance_better_overall = np.logical_and(
    nuance_better_true_positives, nuance_better_false_positives
)

stats["nuance_best_TP_tau<5"] = float(nuance_better_true_positives.mean())
stats["nuance_best_FP_tau<5"] = float(nuance_better_false_positives.mean())
stats["nuance_best_overall_tau<5"] = float(nuance_better_overall.mean())

mask = np.ones_like(tau).astype(bool)
nuance_better_true_positives = tps[main_method][mask] >= others_true_positives[mask]
nuance_better_false_positives = fps[main_method][mask] <= others_false_positives[mask]
nuance_better_overall = np.logical_and(
    nuance_better_true_positives, nuance_better_false_positives
)

stats["nuance_best_TP"] = float(nuance_better_true_positives.mean())
stats["nuance_best_FP"] = float(nuance_better_false_positives.mean())
stats["nuance_best_overall"] = float(nuance_better_overall.mean())

print(nuance_better_true_positives.mean())
print(nuance_better_false_positives.mean())
print(nuance_better_overall.mean())

for method in methods:
    stats[f"{method}_TP"] = float(np.mean(tps[method][detectables[method]]))

yaml.safe_dump(stats, open(snakemake.output[0], "w"))