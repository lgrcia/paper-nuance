import matplotlib.pyplot as plt
import pandas as pd
from nuance.star import Star

data = pickle.load(open(snakemake.input.fluxes, "rb"))
info = yaml.safe_load(open(snakemake.input.info, "r"))
trend = pickle.load(open(snakemake.input.trend, "rb"))["trend"]


methods = {
    "bspline": r"bspline + TLS",
    "wotan3D": r"wotan$_{3D}$ + TLS",
    "harmonics": r"harmonics + TLS",
    "nuance": r"nuance",
}
star = Star(
    info["star_radius"], info["star_mass"], info["star_amplitude"], info["star_period"]
)
sec_color = "0.7"

# grid of plot with ratio 1:3
fig, axes = plt.subplots(
    nrows=len(methods), ncols=2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(8.5, 10)
)
for i, (name, title) in enumerate(methods.items()):
    ax = axes[i, 1]
    df = pd.read_csv(f"../data/140212114/recovered/{name}/results.csv")
    trend = pickle.load(open(f"../data/140212114/recovered/{name}/0.params", "rb"))[
        "trend"
    ]
    df["found"] = df.apply(
        lambda row: right_candidate(
            row["t0"], row["period"], row["true_t0"], row["true_period"]
        ),
        axis=1,
    )
    radius, period, found, tau, delta = df[
        ["radius", "true_period", "found", "tau", "delta"]
    ].values.T

    # setting up ranges
    taus_range = np.min(tau), np.max(tau)
    deltas_range = np.min(delta), np.max(delta)
    periods_range = np.min(periods), np.max(periods)
    radii_range = np.min(radius), np.max(radius)
    extent = (*periods_range, *radii_range)

    ax.imshow(
        found.reshape((10, 10)).astype(bool),
        extent=[*periods_range, *radii_range],
        aspect="auto",
        cmap="Greys_r",
        origin="lower",
    )

    secax = ax.secondary_yaxis("right")
    radii_ticks = ax.get_yticks()
    delta_ticks = star.radius2delta(radii_ticks)
    secax.set_yticks(radii_ticks, [f"{t:.1f}" for t in delta_ticks])
    secax.set_ylabel(r"$\delta$", color=sec_color)
    secax.tick_params(axis="y", colors=sec_color)

    secax = ax.secondary_xaxis("top")
    period_ticks = ax.get_xticks()
    tau_ticks = star.period2tau(period_ticks)
    secax.set_xticks(period_ticks, [f"{t:.1f}" for t in tau_ticks])
    secax.set_xlabel(r"$\tau$", color=sec_color)
    secax.tick_params(axis="x", colors=sec_color)
    ax.set_ylabel("radius (R$_\odot$)")

    ax = axes[i, 0]
    ax.plot(data["time"], data["flux"], ".", c="0.8")
    if trend is not None:
        ax.plot(data["time"], trend, c="k")
    ax.set_xlim(1, 2)
    ax.set_ylabel("diff. flux")
    ax.set_title(title, loc="left")

axes[-1, 1].set_xlabel("period (days)")
axes[-1, 0].set_xlabel(f"time - {info['time0']:.2f} ${info['time0_format']}$")

plt.tight_layout()
plt.tight_layout()

plt.savefig(snakemake.output[0])
