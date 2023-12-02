import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.figure(figsize=(9, 2.5))

nuance = np.array([yaml.safe_load(open(f))["linear"] for f in snakemake.input.nuance])
bls = np.array([yaml.safe_load(open(f))["biweight"] for f in snakemake.input.bls])
points = np.array(snakemake.params.points)

plt.subplot(121)
plt.plot(points, nuance, ".-", c="k", label="linear search (12 cores)")
i = np.flatnonzero(points == 10000)[0]
plt.plot(
    points[i:],
    nuance[i] * (points[i:] / 10000),
    ".-",
    label="N x linear search",
    c="0.5",
    zorder=-1,
)

plt.plot(points, bls, ".-", c="C0", label="biweight")

plt.xlabel("number of points")
plt.ylabel("processing time (s)")

plt.yscale("log")
plt.xscale("log")
plt.ylim(2e-2, 550)

plt.legend()

nuance = [yaml.safe_load(open(f)) for f in snakemake.input.nuance]
nuance = [n["all"] - n["linear"] for n in nuance]
bls = [yaml.safe_load(open(f)) for f in snakemake.input.bls]
bls = [b["bls"] - b["biweight"] for b in bls]
points = np.array(snakemake.params.points)

plt.subplot(122)
plt.plot(points, nuance, ".-", c="k", label="periodic search (12 cores)")
plt.plot(points, bls, ".-", c="C0", label="BLS")

plt.xlabel("number of points")
plt.ylabel("processing time (s)")

plt.yscale("log")
plt.xscale("log")
plt.ylim(2e-2, 550)

plt.legend()
plt.tight_layout()

plt.savefig(snakemake.output[0])