import csv
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from prepare_data import prepare_data
from neural_network import forward
from backpropagation import backward


RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = 100
BATCH_SIZE = 32


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    X_train, _, _, _, Y_train, _ = prepare_data()

    X_batch = X_train[:BATCH_SIZE]
    Y_batch = Y_train[:BATCH_SIZE]

    with (RESULTS / "training_summary.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        experiments = list(csv.DictReader(file))

    raw_rows = []
    summary_rows = []

    for experiment in experiments:
        width = int(experiment["width"])
        learning_rate = float(experiment["learning_rate"])
        model_path = RESULTS / experiment["model_file"]

        with np.load(model_path, allow_pickle=False) as saved:
            parameters = {
                name: saved[name].copy()
                for name in ("W1", "b1", "W2", "b2", "W3", "b3")
            }

        warmup_probabilities, reference_cache = forward(
            X_batch, parameters
        )
        warmup_gradients = backward(
            parameters, reference_cache, Y_batch
        )
        del warmup_probabilities, warmup_gradients

        forward_times = []
        backward_times = []

        for run in range(1, RUNS + 1):
            start = perf_counter_ns()
            probabilities, measured_cache = forward(
                X_batch, parameters
            )
            forward_ms = (perf_counter_ns() - start) / 1_000_000

            del probabilities, measured_cache

            start = perf_counter_ns()
            gradients = backward(
                parameters, reference_cache, Y_batch
            )
            backward_ms = (perf_counter_ns() - start) / 1_000_000

            del gradients

            forward_times.append(forward_ms)
            backward_times.append(backward_ms)

            raw_rows.append({
                "width": width,
                "learning_rate": learning_rate,
                "batch_size": BATCH_SIZE,
                "run": run,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
            })

        forward_median = median(forward_times)
        backward_median = median(backward_times)
        ratio = backward_median / forward_median

        summary_rows.append({
            "width": width,
            "learning_rate": learning_rate,
            "batch_size": BATCH_SIZE,
            "measured_runs_per_operation": RUNS,
            "warmup_runs_per_operation": 1,
            "forward_median_ms": forward_median,
            "backward_median_ms": backward_median,
            "backward_over_forward": ratio,
            "dtype": str(X_batch.dtype),
        })

        print(
            f"\nwidth={width} | learning_rate={learning_rate}"
        )
        print(f"Forward median: {forward_median:.6f} ms")
        print(f"Backward median: {backward_median:.6f} ms")
        print(f"Backward / forward: {ratio:.3f}")

        del reference_cache, parameters

    save_csv(RESULTS / "timing_raw.csv", raw_rows)
    save_csv(RESULTS / "timing_summary.csv", summary_rows)

    notes = (
        f"Platform: {platform.platform()}\n"
        f"Python: {platform.python_version()}\n"
        f"NumPy: {np.__version__}\n"
        f"Input dtype: {X_batch.dtype}\n"
        f"Batch size: {BATCH_SIZE}\n"
        "Data: first 32 training samples from the fixed split.\n"
        "Execution: NumPy on CPU, using default library threading.\n"
        "Models: saved parameters after 50 training epochs.\n"
        "One warm-up per operation, followed by 100 measured runs.\n"
        "Forward and backward were timed separately in each run.\n"
        "Reported statistic: median milliseconds per batch.\n"
        "Forward includes probabilities and intermediate cache creation.\n"
        "Backward uses a fixed precomputed cache for the same model "
        "and batch; it includes the loss gradient calculation.\n"
        "Model loading, data preparation, scalar loss evaluation, "
        "weight updates, printing, and file saving are excluded.\n"
        "Returned outputs are released outside the timed intervals.\n"
        "Parameters remain unchanged during measurement.\n"
        "These are repeated in-process measurements; cache effects "
        "and operating-system scheduling can affect the results.\n"
    )

    (RESULTS / "timing_method.txt").write_text(
        notes,
        encoding="utf-8",
    )

    print("\nTiming results saved.")


if __name__ == "__main__":
    main()
