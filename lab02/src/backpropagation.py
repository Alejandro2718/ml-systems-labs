import csv
from pathlib import Path

import numpy as np

from prepare_data import prepare_data
from neural_network import initialize_parameters, forward, cross_entropy


def backward(parameters, cache, Y):
    batch_size = Y.shape[0]

    dZ3 = (cache["probabilities"] - Y) / batch_size
    dW3 = cache["A2"].T @ dZ3
    db3 = dZ3.sum(axis=0)

    dA2 = dZ3 @ parameters["W3"].T
    dZ2 = dA2 * (cache["Z2"] > 0)
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ parameters["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)

    gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }

    return gradients


def check_gradients(width, X, Y):
    parameters = initialize_parameters(width)
    _, cache = forward(X, parameters)
    gradients = backward(parameters, cache, Y)

    for name, gradient in gradients.items():
        assert gradient.shape == parameters[name].shape
        assert np.all(np.isfinite(gradient))

    epsilon = 1e-5
    threshold = 1e-6
    rng = np.random.default_rng(42)
    rows = []

    for name, count in (("W1", 4), ("W2", 3), ("W3", 3)):
        selected = rng.choice(
            parameters[name].size,
            size=count,
            replace=False,
        )

        for flat_index in selected:
            index = np.unravel_index(
                flat_index,
                parameters[name].shape,
            )
            original = float(parameters[name][index])

            try:
                parameters[name][index] = original + epsilon
                _, plus_cache = forward(X, parameters)
                loss_plus = cross_entropy(plus_cache["log_probs"], Y)

                parameters[name][index] = original - epsilon
                _, minus_cache = forward(X, parameters)
                loss_minus = cross_entropy(minus_cache["log_probs"], Y)
            finally:
                parameters[name][index] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = float(gradients[name][index])

            denominator = max(
                1e-12,
                abs(analytical) + abs(numerical),
            )
            relative_error = abs(analytical - numerical) / denominator
            passed = bool(relative_error < threshold)

            relu_boundary_crossed = any(
                np.any(
                    (plus_cache[key] > 0) != (minus_cache[key] > 0)
                )
                for key in ("Z1", "Z2")
            )

            rows.append({
                "width": width,
                "parameter": name,
                "row": int(index[0]),
                "column": int(index[1]),
                "analytical_gradient": analytical,
                "numerical_gradient": numerical,
                "relative_error": relative_error,
                "epsilon": epsilon,
                "threshold": threshold,
                "relu_boundary_crossed": bool(relu_boundary_crossed),
                "passed": passed,
            })

            status = "PASS" if passed else "FAIL"
            print(
                f"{name}[{index[0]}, {index[1]}] | "
                f"analytical={analytical:.8e} | "
                f"numerical={numerical:.8e} | "
                f"relative_error={relative_error:.3e} | {status}"
            )

    return rows


def main():
    X_train, _, _, _, Y_train, _ = prepare_data()
    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    all_rows = []
    summaries = []

    for width in (16, 64):
        print(f"\nChecking hidden width {width}...")
        rows = check_gradients(width, X_batch, Y_batch)
        passed_count = sum(row["passed"] for row in rows)

        all_rows.extend(rows)
        summaries.append({
            "width": width,
            "checked_weights": len(rows),
            "passed_weights": passed_count,
            "required_passes": 9,
            "lab_requirement_met": passed_count >= 9,
        })

        print(f"Passed: {passed_count}/10")
        print(f"Lab requirement met: {passed_count >= 9}")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for filename, rows in (
        ("gradient_check.csv", all_rows),
        ("gradient_check_summary.csv", summaries),
    ):
        output = results_dir / filename
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"Results saved to: {output}")

    if not all(row["lab_requirement_met"] for row in summaries):
        raise RuntimeError(
            "Gradient checking failed. Inspect the saved results "
            "before training."
        )

    print("\nBoth networks passed the gradient check.")


if __name__ == "__main__":
    main()
