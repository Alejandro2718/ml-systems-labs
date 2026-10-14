import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def prepare_data():
    # Apply the seeds required by the lab manual.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Load the images and scale pixel values to [0, 1].
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0

    # Use the same stratified split for every experiment.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    # Convert integer labels to one-hot vectors.
    Y_train = np.eye(10, dtype=np.float64)[y_train]
    Y_test = np.eye(10, dtype=np.float64)[y_test]

    return X_train, X_test, y_train, y_test, Y_train, Y_test


def main():
    X_train, X_test, y_train, y_test, Y_train, Y_test = prepare_data()

    # Check the input and target formats before building the network.
    assert X_train.shape[1] == 64
    assert X_test.shape[1] == 64
    assert Y_train.shape == (len(y_train), 10)
    assert Y_test.shape == (len(y_test), 10)
    assert np.all((X_train >= 0) & (X_train <= 1))
    assert np.all((X_test >= 0) & (X_test <= 1))
    assert np.all(Y_train.sum(axis=1) == 1)
    assert np.all(Y_test.sum(axis=1) == 1)
    assert np.array_equal(Y_train.argmax(axis=1), y_train)
    assert np.array_equal(Y_test.argmax(axis=1), y_test)

    summary = {
        "dataset": "sklearn.datasets.load_digits",
        "total_samples": len(y_train) + len(y_test),
        "features": X_train.shape[1],
        "classes": 10,
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "scaling": "Pixel values divided by 16",
        "split": "70/30 stratified, random_state=42",
        "training_class_counts": np.bincount(
            y_train, minlength=10
        ).tolist(),
        "test_class_counts": np.bincount(
            y_test, minlength=10
        ).tolist(),
    }

    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(parents=True, exist_ok=True)

    output = results / "dataset_summary.json"
    output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training inputs: {X_train.shape}")
    print(f"Test inputs: {X_test.shape}")
    print(f"Training one-hot labels: {Y_train.shape}")
    print(f"Test one-hot labels: {Y_test.shape}")
    print("All data checks passed.")
    print(f"Summary saved to: {output}")


if __name__ == "__main__":
    main()
