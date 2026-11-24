from statistics import median
from time import perf_counter_ns

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import (DATA, RESULTS, SEED, eval_loader, lab07_choices, latency_row, load_data, load_lab07,
                    save_csv, setup)

TOLERANCE = 1e-4
THREADS = 4


def export(model, path):
    torch.onnx.export(model.eval(), torch.zeros(1, 1, 28, 28), path, opset_version=17,
                      input_names=['images'], output_names=['logits'],
                      dynamic_axes={'images': {0: 'batch'}, 'logits': {0: 'batch'}})
    onnx.checker.check_model(onnx.load(path))


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def ort_latency(runner, batch, runs, warmups=5):
    x = torch.randn(batch, 1, 28, 28, generator=torch.Generator().manual_seed(SEED)).numpy()
    for _ in range(warmups):
        runner.run(None, {'images': x})
    measured = []
    for _ in range(runs):
        start = perf_counter_ns()
        runner.run(None, {'images': x})
        measured.append((perf_counter_ns() - start) / 1e6)
    return median(measured), float(np.percentile(measured, 95))


@torch.inference_mode()
def main():
    setup()
    _, _, test = load_data()
    rows = []
    baseline_name, _, distilled_name = lab07_choices()
    for label, name in [('baseline', baseline_name), ('distilled', distilled_name)]:
        model = load_lab07(name)
        path = DATA / f'{label}.onnx'
        export(model, path)
        runner = session(path)
        max_diff = 0.0
        agree = count = correct_torch = correct_onnx = 0
        for images, labels in eval_loader(test):
            expected = model(images).numpy()
            actual = runner.run(None, {'images': images.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(expected - actual).max()))
            agree += int((expected.argmax(1) == actual.argmax(1)).sum())
            correct_torch += int((expected.argmax(1) == labels.numpy()).sum())
            correct_onnx += int((actual.argmax(1) == labels.numpy()).sum())
            count += len(labels)
        if max_diff > TOLERANCE:
            raise AssertionError(f'ONNX output differs by {max_diff}')
        torch_times = latency_row(model)
        b1, b1_p95 = ort_latency(runner, 1, 300)
        b128, b128_p95 = ort_latency(runner, 128, 40)
        rows.append(dict(model=label, source=name, opset=17, onnxruntime=ort.__version__,
                         provider='CPUExecutionProvider', threads=THREADS, onnx_bytes=path.stat().st_size,
                         test_images=count, max_abs_logit_difference=max_diff, tolerance=TOLERANCE,
                         within_tolerance=max_diff <= TOLERANCE, prediction_agreement=agree / count,
                         pytorch_test_accuracy=correct_torch / count, onnx_test_accuracy=correct_onnx / count,
                         pytorch_batch1_median_ms=torch_times['batch1_median_ms'], onnx_batch1_median_ms=b1,
                         onnx_batch1_p95_ms=b1_p95,
                         pytorch_batch128_median_ms=torch_times['batch128_median_ms'], onnx_batch128_median_ms=b128,
                         onnx_batch128_p95_ms=b128_p95))
        print(f"{label}: max |diff|={max_diff:.2e}, ORT batch1={b1:.3f} ms vs PyTorch "
              f"{torch_times['batch1_median_ms']:.3f} ms", flush=True)
    save_csv(RESULTS / 'onnx_validation.csv', rows)


if __name__ == '__main__':
    main()
