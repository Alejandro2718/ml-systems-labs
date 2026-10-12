import random
import warnings
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

raw_results = []
summary = []
warning_log = []


def fit_model(model, name, phase):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = perf_counter()
        model.fit(X_train, y_train)
        elapsed_ms = (perf_counter() - start) * 1000

    for warning in caught:
        warning_log.append(
            f"{name} | {phase} | "
            f"{warning.category.__name__}: {warning.message}"
        )

    return elapsed_ms


for name, factory in factories.items():
    print(f"\n{name}: running one warm-up...", flush=True)

    warmup_model = factory()
    fit_model(warmup_model, name, "warm-up")
    del warmup_model

    times = []

    for run in range(1, 6):
        model = factory()
        elapsed_ms = fit_model(model, name, f"run {run}")
        del model

        times.append(elapsed_ms)
        raw_results.append({
            "model": name,
            "run": run,
            "training_ms": elapsed_ms,
        })

        print(f"Run {run}/5: {elapsed_ms:.3f} ms", flush=True)

    median_ms = median(times)
    summary.append({
        "model": name,
        "training_median_ms": median_ms,
        "measured_runs": 5,
        "warmup_runs": 1,
    })

    print(f"Median training time: {median_ms:.3f} ms")

pd.DataFrame(raw_results).to_csv(
    results_dir / "training_times_raw.csv",
    index=False,
    float_format="%.6f",
)

pd.DataFrame(summary).to_csv(
    results_dir / "training_time_summary.csv",
    index=False,
    float_format="%.6f",
)

(results_dir / "training_benchmark_warnings.txt").write_text(
    ("\n\n".join(warning_log) or "No training warnings.") + "\n",
    encoding="utf-8",
)

print("\nTraining measurements saved.")
