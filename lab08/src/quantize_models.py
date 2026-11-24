import copy

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping, quantize_dynamic
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from common import (BACKEND, CALIBRATION_IMAGES, RESULTS, SEED, eval_loader, evaluate, lab07_choices,
                    latency_row, load_data, load_lab07, save_csv, setup, state_bytes)

EXAMPLE = (torch.zeros(1, 1, 28, 28),)


def dynamic(model):
    return quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


def calibration_sets(train_set):
    generator = torch.Generator().manual_seed(SEED)
    random_ids = torch.randperm(len(train_set), generator=generator)[:CALIBRATION_IMAGES].tolist()
    images, labels = zip(*(train_set[i] for i in range(len(train_set))))
    images, labels = torch.stack(images), torch.tensor(labels)
    sets = {'random_training_500': images[random_ids]}
    sets['single_class_500'] = images[labels.eq(0)][:CALIBRATION_IMAGES]
    sets['gaussian_noise_500'] = torch.randn(CALIBRATION_IMAGES, 1, 28, 28, generator=generator)
    sets['blank_images_500'] = torch.full((CALIBRATION_IMAGES, 1, 28, 28), -0.2860 / 0.3530)
    return sets, random_ids


@torch.inference_mode()
def static(model, calibration):
    prepared = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping(BACKEND), EXAMPLE)
    for batch in calibration.split(100):
        prepared(batch)
    return convert_fx(prepared)


def describe(name, model, source, method, test, calibration=''):
    row = dict(variant=name, source_model=source, method=method, backend=BACKEND if method != 'fp32' else '',
               calibration=calibration, test_accuracy=evaluate(model, eval_loader(test))[1],
               state_dict_bytes=state_bytes(model))
    row.update(latency_row(model))
    print(f"{name}: accuracy={row['test_accuracy']:.4f} size={row['state_dict_bytes']:,} B "
          f"batch1={row['batch1_median_ms']:.3f} ms", flush=True)
    return row


def main():
    setup()
    train_set, _, test = load_data()
    baseline_name, pruned_name, distilled_name = lab07_choices()
    baseline, pruned, distilled = map(load_lab07, (baseline_name, pruned_name, distilled_name))
    sets, ids = calibration_sets(train_set)
    calibration = sets['random_training_500']
    save_csv(RESULTS / 'calibration_indices.csv', [dict(training_subset_index=i) for i in ids])
    variants = [
        describe('baseline', baseline, baseline_name, 'fp32', test),
        describe('pruned', pruned, pruned_name, 'fp32', test),
        describe('distilled', distilled, distilled_name, 'fp32', test),
        describe('dynamic_quant', dynamic(baseline), baseline_name, 'dynamic_int8_linear', test),
        describe('static_quant', static(baseline, calibration), baseline_name, 'static_int8', test, 'random_training_500'),
        describe('pruned_dynamic_quant', dynamic(pruned), pruned_name, 'dynamic_int8_linear', test),
        describe('pruned_static_quant', static(pruned, calibration), pruned_name, 'static_int8', test, 'random_training_500'),
        describe('distilled_dynamic_quant', dynamic(distilled), distilled_name, 'dynamic_int8_linear', test),
        describe('distilled_static_quant', static(distilled, calibration), distilled_name, 'static_int8', test,
                 'random_training_500'),
    ]
    save_csv(RESULTS / 'variant_comparison.csv', variants)
    calibration_rows = []
    for label, images in sets.items():
        model = static(baseline, images)
        calibration_rows.append(dict(calibration=label, images=len(images), mean=float(images.mean()),
                                     std=float(images.std()), test_accuracy=evaluate(model, eval_loader(test))[1]))
        print(f"Calibration {label}: {calibration_rows[-1]['test_accuracy']:.4f}", flush=True)
    save_csv(RESULTS / 'calibration_study.csv', calibration_rows)


if __name__ == '__main__':
    main()
