import random
import warnings
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

results_dir = Path(__file__).resolve().parents[1] / "results"
results_dir.mkdir(parents=True, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

factories = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "RandomForestClassifier": lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

sample = X_test[:1]

raw_results = []
summary = []
warning_log = []

for name, factory in factories.items():
    print(f"\nPreparing {name}...", flush=True)
    model = factory()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)

    for warning in caught:
        warning_log.append(
            f"{name} | {warning.category.__name__}: {warning.message}"
        )

    accuracy = accuracy_score(y_test, model.predict(X_test))

    model.predict(sample)

    times_ms = []

    for run in range(1, 101):
        start = perf_counter_ns()
        model.predict(sample)
        elapsed_ms = (perf_counter_ns() - start) / 1_000_000

        times_ms.append(elapsed_ms)
        raw_results.append({
            "model": name,
            "run": run,
            "inference_ms": elapsed_ms,
        })

    median_ms = median(times_ms)

    model_dir = results_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"

    joblib.dump(model, model_path, compress=0)
    size_bytes = model_path.stat().st_size

    size_kb = size_bytes / 1000

    summary.append({
        "model": name,
        "accuracy": accuracy,
        "inference_median_ms": median_ms,
        "measured_runs": 100,
        "warmup_runs": 1,
        "model_size_bytes": size_bytes,
        "model_size_KB": size_kb,
        "model_file": model_path.relative_to(results_dir).as_posix(),
    })

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Median single-sample inference: {median_ms:.6f} ms")
    print(f"Model size: {size_bytes} bytes ({size_kb:.3f} KB)")
    print(f"Model saved to: {model_path}")

    del model

pd.DataFrame(raw_results).to_csv(
    results_dir / "inference_times_raw.csv",
    index=False,
    float_format="%.9f",
)

pd.DataFrame(summary).to_csv(
    results_dir / "inference_size_summary.csv",
    index=False,
    float_format="%.9f",
)

(results_dir / "inference_setup_warnings.txt").write_text(
    ("\n\n".join(warning_log) or "No training warnings.") + "\n",
    encoding="utf-8",
)

print("\nInference and model size measurements saved.")
