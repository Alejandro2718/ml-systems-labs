import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS
from models import MODEL_NAMES


def read(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    train = {r["model"]: r for r in read("training_summary.csv")}
    cost = {r["model"]: r for r in read("flops_comparison.csv")}
    timing = {(r["model"], int(r["batch_size"])): r for r in read("timing_summary.csv")}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, ax in enumerate(axes):
        x_values = []
        for name in MODEL_NAMES:
            y = 100 * float(train[name]["test_accuracy"])
            x = int(cost[name]["parameters"]) if i == 0 else float(timing[name, 1 if i == 1 else 128]["median_batch_ms"])
            x_values.append(x)
            ax.scatter(x, y, s=65)
            ax.annotate(name, (x, y), xytext=(5, 6), textcoords="offset points")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlabel(["Trainable parameters", "Median latency, batch 1 (ms)", "Median latency, batch 128 (ms)"][i])
        ax.margins(x=0.25, y=0.25)
        ax.set_xlim(0, max(x_values) * 1.35)
        ax.grid(alpha=0.25)
    fig.suptitle("Fashion-MNIST: final epoch 5, CPU float32, seed 42")
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff_plots.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
