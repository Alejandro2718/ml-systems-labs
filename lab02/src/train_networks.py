import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import prepare_data
from neural_network import initialize_parameters, forward, cross_entropy
from backpropagation import backward


WIDTHS = (16, 64)
LEARNING_RATES = (0.001, 0.01, 0.1)
EPOCHS = 50
BATCH_SIZE = 32
SEED = 42

RESULTS = Path(__file__).resolve().parents[1] / "results"


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(width, learning_rate, X_train, Y_train, y_train):
    # Reset initialization and shuffle order for each experiment.
    parameters = initialize_parameters(width, seed=SEED)
    shuffle_rng = np.random.default_rng(SEED)
    history = []

    for epoch in range(1, EPOCHS + 1):
        order = shuffle_rng.permutation(len(X_train))

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            X_batch = X_train[indices]
            Y_batch = Y_train[indices]

            _, cache = forward(X_batch, parameters)
            gradients = backward(parameters, cache, Y_batch)

            # Mini-batch gradient descent, without momentum.
            for name in parameters:
                parameters[name] -= learning_rate * gradients[name]

        # Evaluate the entire training set after the epoch's updates.
        probabilities, cache = forward(X_train, parameters)
        train_loss = cross_entropy(cache["log_probs"], Y_train)
        train_accuracy = float(
            np.mean(probabilities.argmax(axis=1) == y_train)
        )

        if not np.isfinite(train_loss):
            raise RuntimeError(
                f"Non-finite loss: width={width}, "
                f"learning_rate={learning_rate}, epoch={epoch}"
            )

        history.append({
            "width": width,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        })

        print(
            f"width={width} | lr={learning_rate} | "
            f"epoch={epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"train_accuracy={train_accuracy:.4f}",
            flush=True,
        )

    return parameters, history


def plot_losses(history):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        sharey=True,
    )

    for ax, width in zip(axes, WIDTHS):
        for learning_rate in LEARNING_RATES:
            rows = [
                row for row in history
                if row["width"] == width
                and row["learning_rate"] == learning_rate
            ]

            ax.plot(
                [row["epoch"] for row in rows],
                [row["train_loss"] for row in rows],
                label=f"Learning rate = {learning_rate}",
            )

        ax.set_title(f"Hidden width = {width}")
        ax.set_xlabel("Epoch")
        ax.set_xlim(1, EPOCHS)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Training cross-entropy")
    fig.suptitle("Training loss after each epoch")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=160)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    models_dir = RESULTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, Y_train, _ = prepare_data()

    all_history = []
    summary = []

    for width in WIDTHS:
        for learning_rate in LEARNING_RATES:
            print(
                f"\nStarting width={width}, "
                f"learning_rate={learning_rate}",
                flush=True,
            )

            parameters, history = train_one(
                width,
                learning_rate,
                X_train,
                Y_train,
                y_train,
            )

            # Evaluate test accuracy only after all 50 epochs.
            test_probabilities, _ = forward(X_test, parameters)
            test_predictions = test_probabilities.argmax(axis=1)
            test_accuracy = float(
                np.mean(test_predictions == y_test)
            )

            model_path = models_dir / (
                f"width_{width}_lr_{learning_rate}.npz"
            )
            np.savez(model_path, **parameters)

            all_history.extend(history)
            summary.append({
                "width": width,
                "learning_rate": learning_rate,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "parameter_count": sum(
                    value.size for value in parameters.values()
                ),
                "final_train_loss": history[-1]["train_loss"],
                "final_train_accuracy": history[-1]["train_accuracy"],
                "test_accuracy": test_accuracy,
                "model_file": model_path.relative_to(
                    RESULTS
                ).as_posix(),
            })

            # Save completed experiments progressively.
            save_csv(RESULTS / "training_history.csv", all_history)
            save_csv(RESULTS / "training_summary.csv", summary)

            print(
                f"Final test accuracy: {test_accuracy:.4f}",
                flush=True,
            )

    plot_losses(all_history)

    print("\nFinal results:")
    for row in summary:
        print(
            f"width={row['width']} | "
            f"lr={row['learning_rate']} | "
            f"test_accuracy={row['test_accuracy']:.4f}"
        )

    print(f"\nResults saved to: {RESULTS}")
    print("Loss curves saved to: loss_curves.png")


if __name__ == "__main__":
    main()
