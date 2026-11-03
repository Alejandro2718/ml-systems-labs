import gc
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch import nn

from common import (BATCH_SIZE, DATA, EPOCHS, LEARNING_RATE, RESULTS, SEED,
                    initial_weights, load_data, save_csv, setup)
from models import ScalarModel, TorchModel, predict


def scalar_loss(model, X, y, indices):
    losses = []
    for index in indices:
        p = model(X[index])
        target = y[index]
        losses.append(-(target * p.log() + (1 - target) * (1 - p).log()))
    return sum(losses) / len(indices)


def fit_scalar(model, X, y, batches):
    parameters = model.parameters()
    for indices in batches:
        loss = scalar_loss(model, X, y, indices)
        loss.backward()
        for parameter in parameters:
            parameter.data -= LEARNING_RATE * parameter.grad


def fit_torch(model, X, y, batches, optimizer):
    for indices in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy(model(X[indices]).ravel(), y[indices])
        loss.backward()
        optimizer.step()


def main():
    setup()
    X_train, X_test, y_train, y_test = load_data()
    weights = initial_weights()
    rng = np.random.default_rng(SEED)
    batches = []
    for _ in range(EPOCHS):
        order = rng.permutation(len(X_train))
        batches.extend([order[start:start + BATCH_SIZE].tolist()
                        for start in range(0, len(order), BATCH_SIZE)])
    tensor_batches = [torch.tensor(x, dtype=torch.int64) for x in batches]
    scalar_X, scalar_y = X_train.tolist(), y_train.tolist()
    tensor_X, tensor_y = torch.from_numpy(X_train), torch.from_numpy(y_train)
    scalar = ScalarModel(weights)
    pytorch = TorchModel(weights)
    scalar_loss(scalar, scalar_X, scalar_y, batches[0]).backward()
    nn.functional.binary_cross_entropy(pytorch(tensor_X[batches[0]]).ravel(), tensor_y[batches[0]]).backward()
    expected = np.concatenate([p.grad.numpy().ravel() for p in pytorch.parameters()])
    actual = np.array([p.grad for p in scalar.parameters()])
    gradient_difference = float(np.max(np.abs(actual - expected)))
    assert gradient_difference < 1e-6
    raw, final_weights = [], {}
    for run in range(6):
        names = ['Scalar', 'PyTorch'] if run % 2 == 0 else ['PyTorch', 'Scalar']
        for name in names:
            gc.collect()
            model = ScalarModel(weights) if name == 'Scalar' else TorchModel(weights)
            optimizer = None if name == 'Scalar' else torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
            start = perf_counter()
            if name == 'Scalar':
                fit_scalar(model, scalar_X, scalar_y, batches)
            else:
                fit_torch(model, tensor_X, tensor_y, tensor_batches, optimizer)
            seconds = perf_counter() - start
            learned = model.weights()
            probabilities = predict(learned, X_train)
            train_loss = float(-np.mean(y_train * np.log(probabilities) + (1 - y_train) * np.log1p(-probabilities)))
            accuracy = float(np.mean((predict(learned, X_test) >= 0.5) == y_test))
            assert np.isfinite(train_loss)
            if name in final_weights:
                for key in learned:
                    np.testing.assert_allclose(learned[key], final_weights[name][key], rtol=0, atol=1e-12)
            final_weights[name] = learned
            if run:
                raw.append(dict(framework=name, run=run, training_seconds=seconds,
                                final_training_bce=train_loss, test_accuracy=accuracy))
            print(f'{name} {"warm-up" if not run else f"run {run}/5"}: {seconds:.3f}s, BCE={train_loss:.6f}, accuracy={accuracy:.4f}', flush=True)
            del model
    parameter_difference = max(float(np.max(np.abs(final_weights['Scalar'][key] - final_weights['PyTorch'][key])))
                               for key in weights)
    assert parameter_difference < 1e-6
    np.testing.assert_array_equal(predict(final_weights['Scalar'], X_test) >= 0.5,
                                  predict(final_weights['PyTorch'], X_test) >= 0.5)
    summary = []
    for name in ['Scalar', 'PyTorch']:
        rows = [row for row in raw if row['framework'] == name]
        summary.append(dict(framework=name, parameters=33, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            learning_rate=LEARNING_RATE, warmups=1, runs=5, dtype='float64',
                            median_training_seconds=median(row['training_seconds'] for row in rows),
                            final_training_bce=rows[-1]['final_training_bce'], test_accuracy=rows[-1]['test_accuracy'],
                            initial_gradient_max_abs_difference=gradient_difference,
                            trained_parameter_max_abs_difference=parameter_difference))
    save_csv(RESULTS / 'training_times_raw.csv', raw)
    save_csv(RESULTS / 'training_summary.csv', summary)
    np.savez(DATA / 'model_weights.npz', **final_weights['PyTorch'])


if __name__ == '__main__':
    main()
