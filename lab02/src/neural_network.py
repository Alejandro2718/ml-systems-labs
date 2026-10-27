import csv
from pathlib import Path

import numpy as np

from prepare_data import prepare_data


def initialize_parameters(width, seed=42):
    rng = np.random.default_rng(seed)

    weight_std = np.sqrt(0.01)

    parameters = {
        "W1": rng.normal(0.0, weight_std, size=(64, width)),
        "b1": np.zeros(width, dtype=np.float64),
        "W2": rng.normal(0.0, weight_std, size=(width, width)),
        "b2": np.zeros(width, dtype=np.float64),
        "W3": rng.normal(0.0, weight_std, size=(width, 10)),
        "b3": np.zeros(10, dtype=np.float64),
    }

    return parameters


def relu(Z):
    return np.maximum(0.0, Z)


def forward(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3

    shifted = Z3 - np.max(Z3, axis=1, keepdims=True)
    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )
    probabilities = np.exp(log_probs)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "log_probs": log_probs,
        "probabilities": probabilities,
    }

    return probabilities, cache


def cross_entropy(log_probs, Y):
    return float(-np.mean(np.sum(Y * log_probs, axis=1)))


def main():
    X_train, _, _, _, Y_train, _ = prepare_data()

    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    reference_loss = float(np.log(10))

    for width in (16, 64):
        parameters = initialize_parameters(width)
        probabilities, cache = forward(X_batch, parameters)
        loss = cross_entropy(cache["log_probs"], Y_batch)

        row_sum_error = float(
            np.max(np.abs(probabilities.sum(axis=1) - 1.0))
        )
        parameter_count = sum(
            value.size for value in parameters.values()
        )

        assert cache["Z1"].shape == (32, width)
        assert cache["A1"].shape == (32, width)
        assert cache["Z2"].shape == (32, width)
        assert cache["A2"].shape == (32, width)
        assert probabilities.shape == (32, 10)
        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0.0)
        assert row_sum_error <= 1e-6
        assert np.isfinite(loss)

        assert abs(loss - reference_loss) < 0.1

        rows.append({
            "width": width,
            "parameter_count": parameter_count,
            "batch_size": len(X_batch),
            "output_classes": probabilities.shape[1],
            "max_row_sum_error": row_sum_error,
            "initial_loss": loss,
            "reference_loss": reference_loss,
            "initialization_variance": 0.01,
            "initialization_std": 0.1,
            "checks_passed": True,
        })

        print(f"\nHidden width: {width}")
        print(f"Trainable parameters: {parameter_count}")
        print(f"Output shape: {probabilities.shape}")
        print(f"Maximum row sum error: {row_sum_error:.3e}")
        print(f"Initial loss: {loss:.6f}")
        print(f"Reference ln(10): {reference_loss:.6f}")
        print("All forward and loss checks passed.")

    output = results_dir / "forward_checks.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nChecks saved to: {output}")


if __name__ == "__main__":
    main()
