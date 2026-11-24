import warnings
from statistics import median
from time import perf_counter_ns

import torch
from torch.nn import functional as F

from common import RESULTS, SEED, SPARSITIES, load_state, save_csv, setup
from models import teacher_cnn


def timed(function, runs, warmups=5):
    for _ in range(warmups):
        function()
    measured = []
    for _ in range(runs):
        start = perf_counter_ns()
        function()
        measured.append((perf_counter_ns() - start) / 1e6)
    return median(measured)


@torch.inference_mode()
def main():
    setup()
    rows = []
    generator = torch.Generator().manual_seed(SEED)
    for sparsity in SPARSITIES:
        name = 'global_0' if sparsity == 0 else f'layerwise_{int(sparsity * 100)}'
        model = load_state(teacher_cnn(), name)
        linear = model[10]
        weight, bias = linear.weight.contiguous(), linear.bias
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sparse = weight.to_sparse_csr()
        for batch, runs in ((1, 500), (128, 100)):
            x = torch.randn(batch, weight.shape[1], generator=generator)
            dense_out = F.linear(x, weight, bias)
            sparse_out = (sparse @ x.T).T + bias
            if not torch.allclose(dense_out, sparse_out, atol=1e-4):
                raise AssertionError('Sparse and dense products differ')
            dense_ms = timed(lambda: F.linear(x, weight, bias), runs)
            sparse_ms = timed(lambda: (sparse @ x.T).T + bias, runs)
            rows.append(dict(layer='Linear 6272->144', model=name, weight_sparsity=sparsity,
                             nonzero_weights=int(torch.count_nonzero(weight)), batch_size=batch,
                             dense_median_ms=dense_ms, csr_median_ms=sparse_ms,
                             csr_over_dense=sparse_ms / dense_ms))
            print(f"{name} batch {batch}: dense {dense_ms:.4f} ms, CSR {sparse_ms:.4f} ms", flush=True)
    save_csv(RESULTS / 'sparse_kernel_latency.csv', rows)


if __name__ == '__main__':
    main()
