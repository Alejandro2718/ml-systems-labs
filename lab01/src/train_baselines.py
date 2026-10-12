import random
import warnings
from pathlib import Path

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

lab_dir = Path(__file__).resolve().parents[1]
results_dir = lab_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print(f"Dataset samples: {X.shape[0]}")
print(f"Features per sample: {X.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}

rows = []
warning_log = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)

    for warning in caught:
        message = (
            f"{name} | {warning.category.__name__}: "
            f"{warning.message}"
        )
        warning_log.append(message)
        print(message)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    rows.append({
        "model": name,
        "accuracy": accuracy,
    })

    print(f"Test accuracy: {accuracy:.4f}")

output = results_dir / "baseline_accuracy.csv"
pd.DataFrame(rows).to_csv(
    output,
    index=False,
    float_format="%.4f",
)

warning_text = (
    "\n\n".join(warning_log)
    if warning_log
    else "No training warnings."
)
(results_dir / "training_warnings.txt").write_text(
    warning_text + "\n",
    encoding="utf-8",
)

print(f"\nAccuracy results saved to: {output}")
