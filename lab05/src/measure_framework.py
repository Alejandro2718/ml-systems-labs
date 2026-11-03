from collections import deque
from statistics import median
from time import perf_counter_ns

import numpy as np
import torch

from common import DATA, RESULTS, load_data, save_csv, setup
from models import TorchModel


def benchmark(functions, runs=100):
    rows = []
    with torch.inference_mode():
        for function in functions.values():
            function()
        for run in range(1, runs + 1):
            names = list(functions) if run % 2 else list(reversed(functions))
            for name in names:
                start = perf_counter_ns()
                result = functions[name]()
                elapsed = (perf_counter_ns() - start) / 1e6
                del result
                rows.append(dict(operation=name, run=run, milliseconds=elapsed))
    return rows


def describe(name, tensor):
    return dict(tensor=name, shape=str(tuple(tensor.shape)), dtype=str(tensor.dtype),
                stride=str(tensor.stride()), contiguous=tensor.is_contiguous(), device=str(tensor.device))


def backward_graph(output):
    root = output.grad_fn
    nodes, indices, queue = [root], {root: 0}, deque([root])
    rows = []
    while queue:
        node = queue.popleft()
        parents = []
        for parent, _ in node.next_functions:
            if parent is not None:
                if parent not in indices:
                    indices[parent] = len(nodes)
                    nodes.append(parent)
                    queue.append(parent)
                parents.append(f'n{indices[parent]}')
        leaf = getattr(node, 'variable', None)
        detail = f' leaf shape={tuple(leaf.shape)}, dtype={leaf.dtype}' if leaf is not None else ''
        rows.append(f'n{indices[node]} {type(node).__name__} -> {", ".join(parents) or "none"}{detail}')
    return '\n'.join(rows)


def main():
    setup()
    weights = dict(np.load(DATA / 'model_weights.npz', allow_pickle=False))
    model = TorchModel(weights).eval()
    _, X_test, _, _ = load_data()
    example = torch.from_numpy(X_test[:1].copy())
    output = model(example.clone().requires_grad_(True)).sum()
    eager_graph = backward_graph(output)
    with torch.no_grad():
        start = perf_counter_ns()
        traced = torch.jit.trace(model, example, check_trace=True)
        trace_ms = (perf_counter_ns() - start) / 1e6
    graph_text = 'Eager autograd graph (sum of model outputs):\n' + eager_graph
    graph_text += '\n\nTraced forward graph:\n' + str(traced.inlined_graph) + '\n'
    (RESULTS / 'computation_graph.txt').write_text(graph_text.rstrip() + '\n', encoding='utf-8')
    raw, summaries, tensors = [], [], []
    maximum_difference = 0.0
    for batch in [1, 128]:
        inputs = torch.from_numpy(X_test[:batch].copy())
        tensors.append(describe(f'inference_input_batch_{batch}', inputs))
        with torch.inference_mode():
            expected, actual = model(inputs), traced(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            maximum_difference = max(maximum_difference, (actual - expected).abs().max().item())
        measured = benchmark({'eager': lambda: model(inputs), 'traced': lambda: traced(inputs)})
        raw.extend(dict(batch_size=batch, **row) for row in measured)
        for mode in ['eager', 'traced']:
            times = [row['milliseconds'] for row in measured if row['operation'] == mode]
            summaries.append(dict(mode=mode, batch_size=batch, warmups=1, runs=len(times),
                                  median_batch_ms=median(times), dtype='float64',
                                  trace_setup_ms=trace_ms, maximum_output_difference=maximum_difference))
    save_csv(RESULTS / 'execution_times_raw.csv', raw)
    save_csv(RESULTS / 'execution_summary.csv', summaries)
    base = torch.randn(512, 1024, dtype=torch.float64)
    view = base[:, ::2]
    contiguous = view.contiguous()
    matrix = torch.randn(512, 64, dtype=torch.float64)
    assert not view.is_contiguous() and contiguous.is_contiguous()
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    torch.testing.assert_close(view, contiguous, rtol=0, atol=0)
    torch.testing.assert_close(view @ matrix, contiguous @ matrix, rtol=1e-12, atol=1e-12)
    tensors.extend([describe('layout_base', base), describe('non_contiguous_view', view),
                    describe('contiguous_copy', contiguous), describe('right_matrix', matrix)])
    for name, parameter in model.named_parameters():
        tensors.append(describe(name, parameter))
    measured = benchmark({'non_contiguous_matmul': lambda: view @ matrix,
                          'contiguous_matmul': lambda: contiguous @ matrix,
                          'copy_then_matmul': lambda: view.contiguous() @ matrix})
    save_csv(RESULTS / 'tensor_layout.csv', tensors)
    save_csv(RESULTS / 'layout_times_raw.csv', measured)
    layout_summary = []
    for name in ['non_contiguous_matmul', 'contiguous_matmul', 'copy_then_matmul']:
        times = [row['milliseconds'] for row in measured if row['operation'] == name]
        layout_summary.append(dict(operation=name, warmups=1, runs=len(times),
                                   median_ms=median(times), input_shape='(512, 512)',
                                   output_shape='(512, 64)', dtype='float64'))
    save_csv(RESULTS / 'layout_summary.csv', layout_summary)
    print(f'Eager and traced outputs match; trace setup took {trace_ms:.3f} ms.', flush=True)
    for row in summaries:
        print(f'{row["mode"]}, batch {row["batch_size"]}: {row["median_batch_ms"]:.6f} ms', flush=True)
    for row in layout_summary:
        print(f'{row["operation"]}: {row["median_ms"]:.6f} ms', flush=True)


if __name__ == '__main__':
    main()
