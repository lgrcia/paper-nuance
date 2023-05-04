import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def is_found(period, true_period):
    period_check = np.abs(period - true_period) < 0.01
    alias_check = np.abs(2 * period - true_period) < 0.01
    alias_check |= np.abs(period / 2 - true_period) < 0.01
    if period_check:
        return 1
    elif alias_check:
        return 0.7
    else:
        return 0


results_files = {
    "TLS": snakemake.input.tls,
    "BLS": snakemake.input.bls,
    "nuance": snakemake.input.nuance,
}


methods = ["TLS", "BLS", "nuance"]
plt.figure(None, (7.5, 2.7))

for i, method in enumerate(methods):
    ax = plt.subplot(1, len(methods), i + 1)
    files = results_files[method]
    results = pd.DataFrame([yaml.safe_load(open(f, "r")) for f in files])
    results["found"] = results.apply(
        lambda row: is_found(row["period"], row["true_period"]), axis=1
    )

    period, depth, found = results[["true_period", "true_depth", "found"]].values.T
    period_range = [period.min(), period.max()]
    depth_range = np.array([depth.min(), depth.max()]) * 1e4
    # plt.scatter(period, depth, c=found, s=10, cmap="Greys_r")

    ax.imshow(
        found.reshape(20, 20),
        extent=[*period_range, *depth_range],
        aspect="auto",
        cmap="Greys_r",
        origin="lower",
    )

    ax.set_xlabel("period (days)")

    if i == 0:
        ax.set_ylabel(r"depth ($10^{-4}$)")

    if method == "nuance":
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="w", edgecolor="0.8"),
                mpatches.Patch(color="0.7"),
                mpatches.Patch(color="k"),
            ],
            labels=["correct", "alias", "wrong"],
            loc="upper left",
            fontsize=10,
            frameon=False,
        )

    ax.set_title(method)

plt.tight_layout()
plt.savefig(snakemake.output[0])
