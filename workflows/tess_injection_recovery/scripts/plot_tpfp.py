from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

methods_names = snakemake.params.methods
methods = methods_names.keys()

n = 60
bins = {"tau": np.linspace(0.1, 15, n), "delta": np.linspace(0, 8, n)}

params_true_positives = defaultdict(dict)
params_false_positives = defaultdict(dict)

for result_file in snakemake.input:
    method = result_file.split("/")[-1].split(".")[0]
    tau, delta, tp, tf, detectable = np.load(result_file)
    detectable = detectable.astype(bool)
    tp = tp.astype(bool)
    fp = tf.astype(bool)

    # tau - TP
    total = np.histogram(tau[np.array(detectable)], bins=bins["tau"])[0]
    params_true_positives["tau"][method] = (
        np.histogram(tau[tp], bins=bins["tau"])[0] / total
    )

    # tau - FP
    total = np.histogram(tau, bins=bins["tau"])[0]
    params_false_positives["tau"][method] = (
        np.histogram(tau[fp], bins=bins["tau"])[0] / total
    )

    # delta - TP
    total = np.histogram(delta[np.array(detectable)], bins=bins["delta"])[0]
    params_true_positives["delta"][method] = (
        np.histogram(delta[tp], bins=bins["delta"])[0] / total
    )

    # delta - FP
    total = np.histogram(delta, bins=bins["delta"])[0]
    params_false_positives["delta"][method] = (
        np.histogram(delta[fp], bins=bins["delta"])[0] / total
    )


other_methods = [method for method in methods if method != "nuance"]

def plot(param, ax, which):
    """Plot the true/false positives for a given parameter.

    Parameters
    ----------
    param : string
        "tau" or "delta"
    ax : plt.Axes
    which : "fp" or "tp"
    """
    metric = {
        "fp": params_false_positives,
        "tp": params_true_positives,
    }[which]
    other_function = np.max if which == "tp" else np.min
    ax.step(
        bins[param][0:-1],
        metric[param]["nuance"] * 100,
        c="k",
        label="nuance",
        where="mid",
    )
    others = other_function([metric[param][method] for method in other_methods], 0)
    ax.step(
        bins[param][0:-1], others * 100, c="C0", label="best of others", where="mid"
    )
    for i, method in enumerate(other_methods):
        ax.step(
            bins[param][0:-1],
            metric[param][method] * 100,
            c="C0",
            alpha=0.1,
            label="other methods" if i == 0 else None,
            where="mid",
        )
    ax.set_xlim(bins[param][0], bins[param][-1])
    ax.set_ylim(0, 110)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))

tau_lim = 5
diff_color = "C3"

ax = plt.subplot(221)
plot("tau", ax, "tp")
plt.title("True and false positives against relative parameters", loc="left")
plt.ylabel("true positives (%)")
# plt.axvline(tau_lim, c="k", alpha=0.5)


ax = plt.subplot(222)
plot("delta", ax, "tp")
plt.ylim(0, 110)
plt.xlim(bins["delta"][0], bins["delta"][-2])


ax = plt.subplot(223)
plot("tau", ax, "fp")
plt.xlabel(r"$\tau$")
plt.ylabel("false positives (%)")
plt.legend(loc="upper right")
# plt.axvline(tau_lim, c="k", alpha=0.5)

ax = plt.subplot(224)
plot("delta", ax, "fp")
plt.xlabel(r"$\delta$")
plt.ylim(0, 110)


plt.tight_layout()
plt.savefig(snakemake.output[0])

a = []
b = []

for method in methods:
    tau, delta, tp, tf, detectable = np.load(f"results/{method}.npy")
    detectable = detectable.astype(bool)
    tp = tp.astype(bool)
    fp = tf.astype(bool)
    a.append(tp.sum() / detectable.sum())
    b.append(fp.sum() / detectable.sum())

plt.figure(figsize=(8, 2.5))
x = 8 * np.arange(len(a))
w = 2
plt.bar(x + w / 2, 100 * np.array(a), w, label="true positives", color="0.7")
plt.bar(x - w / 2, 100 * np.array(b), w, color="0.2", label="false positives")
for i, (a_, b_) in enumerate(zip(a, b)):
    plt.text(x[i] + w / 2, 100 * a_ + 5, f"{100*a_:.0f}%", ha="center", fontsize=10)
    plt.text(x[i] - w / 2, 100 * b_ + 5, f"{100*b_:.0f}%", ha="center", fontsize=10)

plt.xticks(x, methods_names.values())
plt.legend(loc="upper left")
plt.ylabel("% of detectable transits")
plt.ylim(0, 100)
plt.title("Overall true and false positives", loc="left")
plt.tight_layout()

plt.savefig(snakemake.output[1])

from scipy.stats import binned_statistic_2d

cmap = "Greys_r"
u = 4
fig = plt.figure(figsize=(8, 3.5))
x = 1e-1
gs = fig.add_gridspec(
    3,
    len(methods),
    height_ratios=[u, u, 1],
    wspace=0.1,
    hspace=0.15,
    left=x / 1.5,
    right=1.0 - x / 3,
    bottom=x,
    top=1.0 - x,
)
for i, f in enumerate(snakemake.input):
    method = f.split("/")[-1].split(".")[0]
    tau, delta, tp, tf, detectable = np.load(f)
    tau, delta, tp, tf = tau.flatten(), delta.flatten(), tp.flatten(), tf.flatten()
    mask = (tau < 12) & (delta < 12)
    bins = 14

    ax = fig.add_subplot(gs[0, i])
    stats = binned_statistic_2d(tau[mask], delta[mask], tp[mask], bins=bins)

    im = ax.imshow(
        stats.statistic.T,
        origin="lower",
        aspect="auto",
        extent=[tau[mask].min(), tau[mask].max(), delta[mask].min(), delta[mask].max()],
        vmax=1,
        vmin=0,
        cmap=cmap,
    )

    if i != 0:
        ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.set_title(methods_names[method], loc="center", fontsize=10)

    ax = fig.add_subplot(gs[1, i])
    stats = binned_statistic_2d(tau[mask], delta[mask], tf[mask], bins=bins)

    im = ax.imshow(
        stats.statistic.T,
        origin="lower",
        aspect="auto",
        extent=[tau[mask].min(), tau[mask].max(), delta[mask].min(), delta[mask].max()],
        vmax=1,
        vmin=0,
        cmap=cmap,
    )

    if i != 0:
        ax.set_yticklabels([])
    ax.set_xlabel(r"$\tau$")

plt.subplot(gs[0, 0]).set_ylabel(r"$\delta$")
plt.subplot(gs[1, 0]).set_ylabel(r"$\delta$")

ax = plt.subplot(gs[0, -1])
ax.text(
    1.1,
    0.5,
    "true positives".upper(),
    ha="center",
    va="center",
    fontsize=10,
    rotation=-90,
    transform=ax.transAxes,
)

ax = plt.subplot(gs[1, -1])
ax.text(
    1.1,
    0.5,
    "false positives".upper(),
    ha="center",
    va="center",
    fontsize=10,
    rotation=-90,
    transform=ax.transAxes,
)

ax = plt.subplot(gs[2, :])
ax.axis("off")
axins = ax.inset_axes((0.5 - 0.15/2, 0 -0.5, 0.15, 0.4))
cb = fig.colorbar(im, cax=axins, orientation="horizontal", ticks=[])
cb.ax.text(-0.05, 0.5, "0%", va="center", ha="right")
cb.ax.text(1.05, 0.5, "100%", va="center", ha="left")

plt.tight_layout()
plt.savefig(snakemake.output[2])