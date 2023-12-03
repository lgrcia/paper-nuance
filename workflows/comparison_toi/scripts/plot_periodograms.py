import pickle
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
plt.subplot(121)
results = pickle.load(open(snakemake.input[0], "rb"))
search = pickle.load(open(snakemake.input[1], "rb"))

plt.plot(search.periods, search.Q_snr, c="C0", label="nuance")
plt.plot(results.period, results.power, c="0.7", label="biweight+BLS", alpha=0.8)
plt.title("First transit search", loc="left")
plt.ylabel("SNR")
plt.xlabel("Period (days)")
y = plt.ylim()

plt.subplot(122)
results = pickle.load(open(snakemake.input[2], "rb"))
search = pickle.load(open(snakemake.input[3], "rb"))

plt.plot(search.periods, search.Q_snr, c="C0", label="nuance")
plt.plot(results.period, results.power, c="0.7", label="biweight+BLS", alpha=0.8)
plt.legend(loc="upper right")
plt.title("Second transit search", loc="left")
plt.xlabel("Period (days)")
plt.ylim(y)

plt.tight_layout()
plt.savefig(snakemake.output[0])
