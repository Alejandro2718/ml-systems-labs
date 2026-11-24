import warnings

import torch
from torch.nn.utils import prune

from common import (FILTER_RATIOS, FINE_TUNE_EPOCHS, FINE_TUNE_LR, RESULTS, SEED, SPARSITIES,
                    clone, eval_loader, evaluate, latency_row, load_data, load_state, save_csv,
                    save_model, setup, size_row, train)
from models import (count_macs, filter_pruned_teacher, nonzero_count, parameter_count,
                    teacher_cnn, weight_layers)


def apply_unstructured(model, method, sparsity):
    layers = [module for _, module in weight_layers(model)]
    if method == 'global':
        prune.global_unstructured([(m, 'weight') for m in layers],
                                  pruning_method=prune.L1Unstructured, amount=sparsity)
    else:
        for module in layers:
            prune.l1_unstructured(module, 'weight', amount=sparsity)


def masks(model):
    return {name: module.weight_mask.clone() for name, module in weight_layers(model)}


def finalize(model):
    for _, module in weight_layers(model):
        prune.remove(module, 'weight')
    return model


def weight_sparsity(model):
    layers = [module.weight for _, module in weight_layers(model)]
    total = sum(w.numel() for w in layers)
    return 1 - sum(int(torch.count_nonzero(w)) for w in layers) / total


def describe(model, validation, test):
    macs, effective = count_macs(model)
    row = dict(parameters=parameter_count(model), nonzero_parameters=nonzero_count(model),
               macs_per_image=macs, nonzero_macs_per_image=effective)
    row['validation_accuracy'] = evaluate(model, eval_loader(validation))[1]
    row['test_accuracy'] = evaluate(model, eval_loader(test))[1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        row.update(size_row(model))
    row.update(latency_row(model))
    return row


def unstructured(baseline, train_set, validation, test):
    rows, layer_rows, histories = [], [], []
    for sparsity in SPARSITIES:
        for method in (['global'] if sparsity == 0 else ['global', 'layerwise']):
            label = f'{method}_{int(sparsity * 100)}'
            model = clone(baseline)
            apply_unstructured(model, method, sparsity)
            before = masks(model)
            accuracy_before = evaluate(model, eval_loader(validation))[1]
            torch.manual_seed(SEED)
            histories += train(model, train_set, FINE_TUNE_EPOCHS, FINE_TUNE_LR, label, validation, seed=SEED + 1)
            if any(not torch.equal(before[name], module.weight_mask) for name, module in weight_layers(model)):
                raise AssertionError('Pruning mask changed during fine-tuning')
            finalize(model)
            for name, module in weight_layers(model):
                if int(torch.count_nonzero(module.weight)) > int(before[name].sum()):
                    raise AssertionError('A pruned weight became non-zero')
                layer_rows.append(dict(run=label, method=method, target_sparsity=sparsity, layer=name,
                                       weights=module.weight.numel(),
                                       sparsity=1 - float(torch.count_nonzero(module.weight)) / module.weight.numel()))
            save_model(model, label)
            row = dict(run=label, method=method, target_sparsity=sparsity,
                       achieved_weight_sparsity=weight_sparsity(model),
                       validation_accuracy_before_fine_tune=accuracy_before)
            row.update(describe(model, validation, test))
            rows.append(row)
            print(f"{label}: sparsity={row['achieved_weight_sparsity']:.4f} test={row['test_accuracy']:.4f}", flush=True)
            if sparsity == 0:
                rows.append(dict(row, run='layerwise_0', method='layerwise'))
    return rows, layer_rows, histories


def structured(baseline, train_set, validation, test):
    rows, histories = [], []
    for ratio in FILTER_RATIOS:
        label = f'filters_{int(ratio * 100)}'
        model, kept = filter_pruned_teacher(baseline, ratio)
        accuracy_before = evaluate(model, eval_loader(validation))[1]
        torch.manual_seed(SEED)
        histories += train(model, train_set, FINE_TUNE_EPOCHS, FINE_TUNE_LR, label, validation, seed=SEED + 1)
        save_model(model, label)
        row = dict(run=label, method='filters_l2', filter_ratio=ratio, kept_filters='/'.join(map(str, kept)),
                   parameter_reduction=1 - parameter_count(model) / parameter_count(baseline),
                   validation_accuracy_before_fine_tune=accuracy_before)
        row.update(describe(model, validation, test))
        rows.append(row)
        print(f"{label}: filters={row['kept_filters']} test={row['test_accuracy']:.4f}", flush=True)
    return rows, histories


def main():
    setup()
    train_set, validation, test = load_data()
    baseline = load_state(teacher_cnn(), 'baseline')
    reference = dict(run='baseline', method='none', target_sparsity=0.0, achieved_weight_sparsity=0.0,
                     validation_accuracy_before_fine_tune=float('nan'))
    reference.update(describe(baseline, validation, test))
    rows, layer_rows, histories = unstructured(baseline, train_set, validation, test)
    save_csv(RESULTS / 'unstructured_pruning.csv', [reference] + rows)
    save_csv(RESULTS / 'unstructured_layer_sparsity.csv', layer_rows)
    filter_rows, filter_histories = structured(baseline, train_set, validation, test)
    save_csv(RESULTS / 'structured_pruning.csv', filter_rows)
    save_csv(RESULTS / 'pruning_histories.csv', histories + filter_histories)


if __name__ == '__main__':
    main()
