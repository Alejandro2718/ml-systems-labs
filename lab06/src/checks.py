import copy
import pandas as pd
import torch
from torch import nn

from common import (DATA, RESULTS, changed, close_loader, make_datasets,
                    make_loader, make_model, make_optimizer, save_csv, setup)
from engine import log_configuration, open_tracking, train_epoch, warmup


def profile_epoch(config):
    setup(config)
    dataset, _, _ = make_datasets(config)
    loader = make_loader(dataset, config)
    model = make_model(config)
    optimizer = make_optimizer(model, config)
    warmup(model, loader, config)
    mlflow = open_tracking(config)
    with mlflow.start_run(run_name='profile_epoch') as run:
        log_configuration(mlflow, config, 'profile_epoch', 'profile')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            measured = train_epoch(model, optimizer, loader, config, 1, profiling=True, memory=False)
        events = profiler.key_averages()
        selected = {event.key:event.cpu_time_total / 1000 for event in events
                    if event.key in ['data_loading', 'forward_pass', 'backward_pass', 'optimizer_step']}
        if len(selected) != 4:
            raise AssertionError('The profiler did not capture all four training stages')
        total_ms = measured['training_seconds'] * 1000
        marked_ms = sum(selected.values())
        if marked_ms > total_ms * 1.01:
            raise AssertionError('Overlapping profile ranges cannot be reported as wall-time shares')
        denominator = max(total_ms, marked_ms)
        rows = [dict(stage=key, milliseconds=value, percent_epoch_wall=100 * value / denominator)
                for key,value in selected.items()]
        rows.append(dict(stage='other_and_profiler_overhead', milliseconds=max(0, total_ms-marked_ms),
                         percent_epoch_wall=100 * max(0, total_ms-marked_ms) / denominator))
        save_csv(RESULTS / 'profile_breakdown.csv', rows)
        operators = [dict(operator=event.key, calls=event.count,
                          self_cpu_ms=event.self_cpu_time_total / 1000,
                          total_cpu_ms=event.cpu_time_total / 1000) for event in events]
        save_csv(RESULTS / 'profile_operators.csv', operators)
        mlflow.log_metrics({f'{row["stage"]}_share_percent':row['percent_epoch_wall'] for row in rows})
        mlflow.log_metric('profile_epoch_seconds', measured['training_seconds'])
        mlflow.log_artifact(str(RESULTS / 'profile_breakdown.csv'))
        mlflow.log_artifact(str(RESULTS / 'profile_operators.csv'))
        print(f'Profiled one complete epoch: {measured["training_seconds"]:.1f}s', flush=True)
        identifier = run.info.run_id
    close_loader(loader)
    return identifier


def check_accumulation(config):
    setup(config)
    dataset, _, _ = make_datasets(config)
    current = changed(config, batch_size=config['studies']['accumulation_reference_batch'],
                      num_workers=0, accumulation_steps=1)
    loader = make_loader(dataset, current)
    images, labels = next(iter(loader))
    model = make_model(current)
    accumulated = copy.deepcopy(model)
    nn.functional.cross_entropy(model(images), labels).backward()
    micro = config['studies']['accumulation_batch']
    for start in range(0, len(labels), micro):
        logits = accumulated(images[start:start+micro])
        (nn.functional.cross_entropy(logits, labels[start:start+micro], reduction='sum') / len(labels)).backward()
    differences = [float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(), accumulated.parameters())]
    maximum = max(differences)
    if maximum >= 1e-6:
        raise AssertionError(f'Accumulated gradients differ by {maximum}')
    reference = pd.read_csv(RESULTS / 'histories/batch_256.csv')
    actual = pd.read_csv(RESULTS / 'histories/accumulation_64x4.csv')
    if not reference.optimizer_steps.equals(actual.optimizer_steps):
        raise AssertionError('Accumulation changed the number of optimizer updates')
    rows = []
    for _,row in reference.iterrows():
        other = actual.loc[actual.epoch.eq(row.epoch)].iloc[0]
        rows.append(dict(epoch=int(row.epoch), batch256_validation_loss=row.validation_loss,
                         accumulated_validation_loss=other.validation_loss,
                         batch256_validation_accuracy=row.validation_accuracy,
                         accumulated_validation_accuracy=other.validation_accuracy,
                         accuracy_difference=abs(row.validation_accuracy-other.validation_accuracy),
                         loss_difference=abs(row.validation_loss-other.validation_loss),
                         optimizer_steps=int(row.optimizer_steps)))
    last = rows[-1]
    passed = (last['accuracy_difference'] <= config['studies']['accumulation_accuracy_tolerance']
              and last['loss_difference'] <= config['studies']['accumulation_loss_tolerance'])
    if not passed:
        raise AssertionError('Accumulation did not meet the predefined behaviour tolerances')
    save_csv(RESULTS / 'accumulation_comparison.csv', rows)
    save_csv(RESULTS / 'accumulation_check.csv', [dict(maximum_single_update_gradient_difference=maximum,
                                                     accuracy_tolerance=config['studies']['accumulation_accuracy_tolerance'],
                                                     loss_tolerance=config['studies']['accumulation_loss_tolerance'], passed=passed)])


def check_resume(config):
    original = torch.load(DATA / 'checkpoints/batch_128.pth', weights_only=False, map_location='cpu')
    resumed = torch.load(DATA / 'checkpoints/resume_check.pth', weights_only=False, map_location='cpu')
    maximum = max(float((original['model'][key]-resumed['model'][key]).abs().max()) for key in original['model'])
    if maximum != 0:
        raise AssertionError(f'Resumed weights differ from uninterrupted weights: {maximum}')
    reference = pd.read_csv(RESULTS / 'histories/batch_128.csv')
    actual = pd.read_csv(RESULTS / 'histories/resume_check.csv')
    columns = ['epoch','train_loss','train_accuracy','learning_rate','validation_loss','validation_accuracy','optimizer_steps']
    pd.testing.assert_frame_equal(reference[columns], actual[columns], check_exact=True)
    rows = []
    for _,row in reference.iterrows():
        other = actual.loc[actual.epoch.eq(row.epoch)].iloc[0]
        rows.append(dict(epoch=int(row.epoch), uninterrupted_validation_loss=row.validation_loss,
                         resumed_validation_loss=other.validation_loss,
                         uninterrupted_validation_accuracy=row.validation_accuracy,
                         resumed_validation_accuracy=other.validation_accuracy,
                         learning_rate=row.learning_rate,
                         interruption_boundary=int(row.epoch)==config['studies']['interrupt_epoch']))
    save_csv(RESULTS / 'resume_comparison.csv', rows)
    save_csv(RESULTS / 'resume_check.csv', [dict(interrupt_epoch=config['studies']['interrupt_epoch'],
                                               completed_epochs=resumed['epoch'],
                                               maximum_parameter_difference=maximum, exact_metric_match=True)])
    print('Resumed and uninterrupted training match exactly.', flush=True)
