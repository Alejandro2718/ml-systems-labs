import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker

from common import (BASELINE_EPOCHS, FINE_TUNE_EPOCHS, RESULTS, STUDENT_EPOCHS, eval_loader,
                    load_data, load_state, save_csv, setup)
from models import filter_pruned_teacher, student_cnn, teacher_cnn

TRAIN_SAMPLES = 55000
TOLERANCE = 0.01
ENERGY_REPEATS = 3
LATENCY_COLUMNS = ['batch1_median_ms', 'batch128_median_ms']


def select(frame, baseline_accuracy, key):
    ok = frame.loc[frame.validation_accuracy >= baseline_accuracy - TOLERANCE]
    return (ok if len(ok) else frame).sort_values(key).iloc[-1]


def interleaved(frame, key, timing):
    frame = frame.copy()
    runs = frame[key].replace({'layerwise_0': 'global_0', 'baseline_cnn': 'baseline'})
    for column in LATENCY_COLUMNS:
        frame[column] = runs.map(timing[column]).fillna(frame[column])
    return frame


def build(name):
    if name.startswith('filters_'):
        shape, _ = filter_pruned_teacher(teacher_cnn(), int(name.split('_')[1]) / 100)
        return load_state(shape, name)
    if name.startswith('student'):
        return load_state(student_cnn(), name)
    return load_state(teacher_cnn(), name)


@torch.inference_mode()
def inference_energy(model, loader):
    logging.getLogger('codecarbon').setLevel(logging.ERROR)
    tracker = OfflineEmissionsTracker(country_iso_code='POL', save_to_file=False, log_level='error',
                                      measure_power_secs=1)
    model.eval()
    images = 0
    tracker.start()
    for _ in range(ENERGY_REPEATS):
        for x, _ in loader:
            model(x)
            images += len(x)
    tracker.stop()
    data = tracker.final_emissions_data
    return dict(energy_joules_per_1000_images=data.energy_consumed * 3.6e6 / images * 1000,
                energy_window_seconds=data.duration, energy_cpu_power_w=data.cpu_power,
                energy_ram_power_w=data.ram_power, energy_tracking=f'codecarbon {data.tracking_mode}, CPU constant mode')


def plots(unstructured, structured, kernels, distillation, baseline):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, marker in [('global', 'o'), ('layerwise', 's')]:
        part = unstructured.loc[unstructured.method.eq(method)].sort_values('target_sparsity')
        axes[0].plot(part.target_sparsity, part.test_accuracy, marker + '-', label=f'{method}, after 3-epoch fine-tune')
        axes[0].plot(part.target_sparsity, part.validation_accuracy_before_fine_tune, marker + ':',
                     alpha=0.6, label=f'{method}, before fine-tune (validation)')
    axes[0].axhline(baseline.test_accuracy, color='grey', linestyle='--', label='Baseline test accuracy')
    axes[0].set(xlabel='Weight sparsity (fraction of weights set to zero)', ylabel='Accuracy',
                title='Unstructured magnitude pruning')
    axes[0].legend(fontsize=7)
    axes[1].plot([0] + list(structured.filter_ratio), [baseline.test_accuracy] + list(structured.test_accuracy),
                 'o-', label='After 3-epoch fine-tune')
    axes[1].plot(structured.filter_ratio, structured.validation_accuracy_before_fine_tune, 'o:', alpha=0.6,
                 label='Before fine-tune (validation)')
    for _, row in structured.iterrows():
        axes[1].annotate(f"{row.parameter_reduction:.0%} fewer params", (row.filter_ratio, row.test_accuracy),
                         textcoords='offset points', xytext=(-30, 8), fontsize=7)
    axes[1].set(xlabel='Fraction of convolution filters removed (L2 norm)', ylabel='Accuracy',
                title='Structured filter pruning')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / 'sparsity_curves.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, marker in [('global', 'o'), ('layerwise', 's')]:
        part = unstructured.loc[unstructured.method.eq(method)].sort_values('target_sparsity')
        axes[0].plot(part.achieved_weight_sparsity, part.batch1_median_ms, marker + '-', label=f'Unstructured {method}')
    axes[0].plot(structured.parameter_reduction, structured.batch1_median_ms, '^-', label='Structured filters')
    axes[0].set(xlabel='Removed weights / parameters (fraction)', ylabel='Batch-1 median latency (ms)',
                title='Full-model PyTorch CPU latency (interleaved)')
    axes[0].legend(fontsize=8)
    for batch in (1, 128):
        part = kernels.loc[kernels.batch_size.eq(batch)]
        axes[1].plot(part.weight_sparsity, part.csr_over_dense, 'o-', label=f'Batch {batch}')
    axes[1].axhline(1, color='grey', linestyle='--')
    axes[1].set_yscale('log')
    axes[1].set(xlabel='Weight sparsity of Linear 6272->144', ylabel='CSR time / dense time',
                title='Sparse CSR kernel vs dense kernel')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / 'latency_analysis.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, part in distillation.groupby('run'):
        ax.plot(part.epoch, part.validation_accuracy, 'o-', label=name)
    ax.set(xlabel='Epoch', ylabel='Validation accuracy', title='Student with and without the teacher')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / 'distillation_curves.png', dpi=160)
    plt.close(fig)


def main():
    setup()
    _, _, test = load_data()
    loader = eval_loader(test)
    timing = pd.read_csv(RESULTS / 'latency_interleaved.csv').set_index('run')
    baseline = interleaved(pd.read_csv(RESULTS / 'baseline_summary.csv'), 'model', timing).iloc[0]
    unstructured = interleaved(pd.read_csv(RESULTS / 'unstructured_pruning.csv'), 'run', timing)
    structured = interleaved(pd.read_csv(RESULTS / 'structured_pruning.csv'), 'run', timing)
    distilled = interleaved(pd.read_csv(RESULTS / 'distillation_summary.csv'), 'model', timing)
    kernels = pd.read_csv(RESULTS / 'sparse_kernel_latency.csv')
    plots(unstructured.loc[unstructured.run.ne('baseline')], structured, kernels,
          pd.read_csv(RESULTS / 'distillation_histories.csv'), baseline)
    pruned = unstructured.loc[unstructured.method.isin(['global', 'layerwise'])]
    chosen = {}
    for method in ['global', 'layerwise']:
        chosen[method] = select(pruned.loc[pruned.method.eq(method)], baseline.validation_accuracy, 'target_sparsity').run
    chosen['filters'] = select(structured, baseline.validation_accuracy, 'filter_ratio').run
    rows = []
    sources = [('baseline', baseline, BASELINE_EPOCHS, 0)]
    for key in ['global', 'layerwise']:
        sources.append((chosen[key], unstructured.set_index('run').loc[chosen[key]], BASELINE_EPOCHS + FINE_TUNE_EPOCHS, 0))
    sources.append((chosen['filters'], structured.set_index('run').loc[chosen['filters']],
                    BASELINE_EPOCHS + FINE_TUNE_EPOCHS, 0))
    for _, row in distilled.iterrows():
        teacher_passes = STUDENT_EPOCHS if row.model == 'student_distilled' else 0
        sources.append((row.model, row, STUDENT_EPOCHS + (BASELINE_EPOCHS if teacher_passes else 0), teacher_passes))
    for name, row, epochs, teacher_epochs in sources:
        model = build(name)
        nonzero = row.get('nonzero_parameters', row.parameters)
        effective = row.get('nonzero_macs_per_image', row.macs_per_image)
        result = dict(variant=name, test_accuracy=row.test_accuracy,
                      data_training_sample_passes=epochs * TRAIN_SAMPLES,
                      data_teacher_forward_passes=teacher_epochs * TRAIN_SAMPLES,
                      model_parameters=int(row.parameters), model_nonzero_parameters=int(nonzero),
                      model_dense_bytes=int(row.dense_bytes), model_gzip_bytes=int(row.dense_gzip_bytes),
                      model_sparse_csr_bytes=int(row.sparse_csr_bytes),
                      compute_macs_per_image=int(row.macs_per_image), compute_nonzero_macs_per_image=int(effective),
                      compute_batch1_median_ms=row.batch1_median_ms, compute_batch128_median_ms=row.batch128_median_ms)
        result.update(inference_energy(model, loader))
        rows.append(result)
        print(f"{name}: {result['energy_joules_per_1000_images']:.2f} J per 1,000 images (estimate)", flush=True)
    save_csv(RESULTS / 'efficiency_summary.csv', rows)


if __name__ == '__main__':
    main()
