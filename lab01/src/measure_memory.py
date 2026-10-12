import argparse
import csv
import gc
import json
import random
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import joblib
import numpy as np
import psutil
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL_NAMES = ["LogisticRegression", "RandomForestClassifier"]


def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def sample_memory(operation):
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    ready = threading.Event()

    def monitor():
        samples.append(process.memory_info().rss)
        ready.set()
        while not stop.wait(0.001):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    ready.wait()

    try:
        operation()
    finally:
        samples.append(process.memory_info().rss)
        stop.set()
        thread.join()

    return baseline, max(samples), len(samples)


def worker(name, phase):
    set_seeds()

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    if phase == "training":
        if name == "LogisticRegression":
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            )

        def operation():
            model.fit(X_train, y_train)

    else:
        model = joblib.load(RESULTS / name / "model.joblib")
        sample = X_test[:1].copy()

        del X, y, X_train, X_test, y_train, y_test

        model.predict(sample)

        def operation():
            for _ in range(100):
                model.predict(sample)

    gc.collect()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        baseline, peak, count = sample_memory(operation)

    warning_text = "\n\n".join(
        f"{item.category.__name__}: {item.message}"
        for item in caught
    )

    (RESULTS / f"memory_{name}_{phase}_warnings.txt").write_text(
        (warning_text or "No operation warnings.") + "\n",
        encoding="utf-8",
    )

    return {
        "model": name,
        "phase": phase,
        "baseline_RSS_MB": baseline / 1_000_000,
        "peak_RSS_bytes": peak,
        "peak_RSS_MB": peak / 1_000_000,
        "observed_increase_MB": (peak - baseline) / 1_000_000,
        "rss_samples": count,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for name in MODEL_NAMES:
        for phase in ["training", "inference"]:
            print(f"Measuring {name}: {phase}...", flush=True)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model",
                    name,
                    "--phase",
                    phase,
                ],
                capture_output=True,
                text=True,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr or completed.stdout
                )

            row = json.loads(completed.stdout)
            rows.append(row)

            print(
                f"Observed peak RSS: {row['peak_RSS_MB']:.3f} MB",
                flush=True,
            )

    output = RESULTS / "memory_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    notes = (
        "Memory method: sampled process RSS using psutil.\n"
        "Each model and phase ran in a fresh Python process.\n"
        "Training: one fit call, including its temporary allocations.\n"
        "Inference: one warm-up followed by 100 single-sample predictions.\n"
        "Inference excludes model loading and releases unused dataset arrays.\n"
        "Requested sampling interval: 1 ms; actual scheduling may vary.\n"
        "Brief memory peaks may be missed between samples.\n"
        "RSS includes Python, imported libraries (including PyTorch), "
        "the model, and other live process allocations.\n"
        "The observed increase is not the total memory requirement.\n"
        "Units: 1 MB = 1000000 bytes.\n"
    )
    (RESULTS / "memory_method.txt").write_text(
        notes,
        encoding="utf-8",
    )

    print(f"\nMemory results saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--phase", choices=["training", "inference"])
    args = parser.parse_args()

    if args.model and args.phase:
        print(json.dumps(worker(args.model, args.phase)))
    else:
        main()
