import numpy as np
import torch

from common import RESULTS, save_csv, setup
from value import Value


def scalar_expression(a, b, c):
    return sum(((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c))


def main():
    setup()
    a = np.array([-1.5, 0.8, 2.0, -0.7])
    b = np.array([0.4, -1.2, 0.5, 1.3])
    c = np.array([0.2, 0.3, -0.4, 0.1])
    inputs = [a, b, c]
    nodes = [[Value(x) for x in values] for values in inputs]
    output = scalar_expression(*nodes)
    output.backward()
    tensors = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    reference = ((tensors[0] * tensors[1] + tensors[2]) ** 2).relu().sum()
    reference.backward()
    np.testing.assert_allclose(output.data, reference.item(), rtol=0, atol=1e-12)
    rows = []
    epsilon = 1e-6
    for group, name in enumerate(['a', 'b', 'c']):
        for index, node in enumerate(nodes[group]):
            plus, minus = [x.copy() for x in inputs], [x.copy() for x in inputs]
            plus[group][index] += epsilon
            minus[group][index] -= epsilon
            function = lambda x: np.maximum((x[0] * x[1] + x[2]) ** 2, 0).sum()
            numerical = (function(plus) - function(minus)) / (2 * epsilon)
            expected = tensors[group].grad[index].item()
            rows.append(dict(variable=name, index=index, scalar_gradient=node.grad,
                             torch_gradient=expected, numerical_gradient=numerical,
                             torch_abs_difference=abs(node.grad - expected),
                             numerical_abs_difference=abs(node.grad - numerical)))
    maximum_torch = max(row['torch_abs_difference'] for row in rows)
    maximum_numeric = max(row['numerical_abs_difference'] for row in rows)
    save_csv(RESULTS / 'gradient_check.csv', rows)
    assert max(maximum_torch, maximum_numeric) < 1e-6
    shared = Value(2)
    expression = shared * shared + shared
    expression.backward()
    assert shared.grad == 5
    expression.backward()
    assert shared.grad == 5
    for value in [-1, 0, 1]:
        x = Value(value)
        x.relu().backward()
        assert x.grad == float(value > 0)
    x = Value(0)
    (x ** 0).backward()
    assert x.grad == 0
    print(f'Gradient checks: torch={maximum_torch:.3e}, numerical={maximum_numeric:.3e}', flush=True)
    print('Shared-parent accumulation, repeated backward, ReLU and zero-power checks passed.', flush=True)


if __name__ == '__main__':
    main()
