import copy
import random
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
import yaml
from torch import nn

from common import (DATA, RESULTS, ROOT, MemoryMonitor, close_loader, config_hash,
                    make_datasets, make_loader, make_model, make_optimizer,
                    make_scheduler, save_csv, setup)


def record(name, enabled):
    return torch.profiler.record_function(name) if enabled else nullcontext()


def train_epoch(model, optimizer, loader, config, epoch, profiling=False, memory=True):
    model.train()
    loader.sampler.epoch = epoch
    physical = config['training']['batch_size']
    effective = physical * config['training']['accumulation_steps']
    total = len(loader.dataset)
    seen = correct = window_seen = updates = 0
    loss_total = wait_seconds = 0.0
    monitor = MemoryMonitor(config['training']['memory_sample_seconds']) if memory else nullcontext()
    with monitor as peak:
        start = perf_counter()
        iterator = iter(loader)
        for _ in range(len(loader)):
            wait_start = perf_counter()
            with record('data_loading', profiling):
                images, labels = next(iterator)
            wait_seconds += perf_counter() - wait_start
            if window_seen == 0:
                window_total = min(effective, total - seen)
                optimizer.zero_grad(set_to_none=True)
            with record('forward_pass', profiling):
                logits = model(images)
                loss_sum = nn.functional.cross_entropy(logits, labels, reduction='sum')
                loss = loss_sum / window_total
            with record('backward_pass', profiling):
                loss.backward()
            window_seen += len(labels)
            if window_seen == window_total:
                with record('optimizer_step', profiling):
                    optimizer.step()
                window_seen = 0
                updates += 1
            loss_total += loss_sum.item()
            correct += int(logits.argmax(1).eq(labels).sum())
            seen += len(labels)
        seconds = perf_counter() - start
    if seen != total or window_seen:
        raise AssertionError('The epoch did not consume every sample exactly once')
    return dict(epoch=epoch, train_loss=loss_total / seen, train_accuracy=correct / seen,
                samples=seen, optimizer_steps=updates, training_seconds=seconds,
                throughput_samples_s=seen / seconds, data_wait_seconds=wait_seconds,
                peak_parent_rss_mb=peak.parent_peak / 1e6 if peak else 0,
                peak_tree_rss_mb=peak.tree_peak / 1e6 if peak else 0)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss = correct = seen = 0
    for images, labels in loader:
        logits = model(images)
        loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        correct += int(logits.argmax(1).eq(labels).sum())
        seen += len(labels)
    return loss / seen, correct / seen


def warmup(model, loader, config):
    state = torch.get_rng_state()
    temporary = copy.deepcopy(model)
    optimizer = make_optimizer(temporary, config)
    images, labels = next(iter(loader))
    loss = nn.functional.cross_entropy(temporary(images), labels)
    loss.backward()
    optimizer.step()
    del temporary, optimizer, images, labels, loss
    torch.set_rng_state(state)


def save_checkpoint(path, model, optimizer, scheduler, loader, config, epoch, history, run_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), epoch=epoch, history=history,
                 config=config, config_sha256=config_hash(config), run_id=run_id,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), loader_rng=loader.generator.get_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def open_tracking(config):
    import mlflow
    mlflow.set_tracking_uri((DATA / 'mlruns').as_uri())
    mlflow.set_experiment(config['tracking']['experiment'])
    return mlflow


def log_configuration(mlflow, config, name, study):
    values = config['training']
    params = dict(seed=config['seed'], model=config['model'], device=config['device'],
                  threads=config['threads'], dataset=config['dataset']['name'],
                  train_samples=config['dataset']['train_samples'],
                  validation_samples=config['dataset']['validation_samples'],
                  test_samples=config['dataset']['test_samples'], config_sha256=config_hash(config),
                  **values)
    mlflow.log_params(params)
    mlflow.set_tags(dict(study=study, run_name=name))
    mlflow.log_text(yaml.safe_dump(config, sort_keys=False), 'config.yaml')


def train(config, name, study, stop_after=None, resume=None):
    setup(config)
    train_data, validation, test = make_datasets(config)
    loader = make_loader(train_data, config)
    val_loader = make_loader(validation, config, training=False)
    test_loader = make_loader(test, config, training=False)
    model = make_model(config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    history = []
    start_epoch = 1
    checkpoint = DATA / 'checkpoints' / f'{name}.pth'
    saved = None
    if resume:
        saved = torch.load(resume, weights_only=False, map_location='cpu')
        if saved['config_sha256'] != config_hash(config):
            raise ValueError('The resume configuration differs from its checkpoint')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history = saved['history']
        start_epoch = saved['epoch'] + 1
    warmup(model, loader, config)
    if saved:
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        loader.generator.set_state(saved['loader_rng'])
        if saved['cuda_rng']:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    mlflow = open_tracking(config)
    run = mlflow.start_run(run_id=saved['run_id'] if saved else None, run_name=name)
    log_configuration(mlflow, config, name, study)
    if saved:
        mlflow.set_tag('resumed_from_epoch', saved['epoch'])
    stopped = False
    try:
        for epoch in range(start_epoch, config['training']['epochs'] + 1):
            learning_rate = optimizer.param_groups[0]['lr']
            row = train_epoch(model, optimizer, loader, config, epoch)
            row['learning_rate'] = learning_rate
            row['validation_loss'], row['validation_accuracy'] = evaluate(model, val_loader)
            if config['training']['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(row['validation_loss'])
            else:
                scheduler.step()
            history.append(row)
            mlflow.log_metrics({key:float(value) for key,value in row.items() if key != 'epoch'}, step=epoch)
            save_checkpoint(checkpoint, model, optimizer, scheduler, loader, config, epoch, history, run.info.run_id)
            print(f'{name} epoch {epoch}/{config["training"]["epochs"]}: val={row["validation_accuracy"]:.4f}, loss={row["validation_loss"]:.4f}, {row["training_seconds"]:.1f}s', flush=True)
            if stop_after and epoch == stop_after:
                import shutil
                interrupted = checkpoint.with_name(f'{name}_epoch{epoch}.pth')
                shutil.copyfile(checkpoint, interrupted)
                mlflow.log_artifact(str(interrupted), 'checkpoints')
                mlflow.set_tag('intentional_stop_epoch', epoch)
                stopped = True
                break
        history_path = RESULTS / 'histories' / f'{name}.csv'
        save_csv(history_path, history)
        mlflow.log_artifact(str(history_path), 'history')
        mlflow.log_artifact(str(checkpoint), 'checkpoints')
        result = dict(name=name, study=study, run_id=run.info.run_id,
                      epochs_completed=history[-1]['epoch'], checkpoint=str(checkpoint.relative_to(ROOT)),
                      validation_loss=history[-1]['validation_loss'], validation_accuracy=history[-1]['validation_accuracy'],
                      stopped=stopped)
        if not stopped:
            result['test_loss'], result['test_accuracy'] = evaluate(model, test_loader)
            mlflow.log_metrics(dict(test_loss=result['test_loss'], test_accuracy=result['test_accuracy']))
        mlflow.end_run(status='KILLED' if stopped else 'FINISHED')
        return result, history
    except Exception:
        mlflow.end_run(status='FAILED')
        raise
    finally:
        close_loader(loader)
