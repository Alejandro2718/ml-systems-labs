import statistics
from time import perf_counter

import torch
from torch.utils.data import Subset

from common import (RESULTS, changed, close_loader, make_datasets, make_loader,
                    make_model, make_optimizer, save_csv, setup)
from engine import log_configuration, open_tracking, train_epoch, warmup


def worker_study(config):
    setup(config)
    dataset, _, _ = make_datasets(config)
    count = min(len(dataset), config['studies']['worker_batches'] * config['training']['batch_size'])
    dataset = Subset(dataset, list(range(count)))
    raw, summary = [], []
    for workers in config['studies']['workers']:
        current = changed(config, num_workers=workers, epochs=1)
        loader = make_loader(dataset, current)
        torch.manual_seed(config['seed'])
        model = make_model(current)
        start = perf_counter()
        warmup(model, loader, current)
        startup = perf_counter() - start
        optimizer = make_optimizer(model, current)
        train_epoch(model, optimizer, loader, current, 1, memory=False)
        mlflow = open_tracking(current)
        with mlflow.start_run(run_name=f'workers_{workers}') as run:
            log_configuration(mlflow, current, f'workers_{workers}', 'workers')
            mlflow.log_params(dict(benchmark_samples=count, repetitions=config['studies']['worker_runs']))
            rows = []
            for repeat in range(1, config['studies']['worker_runs'] + 1):
                torch.manual_seed(config['seed'])
                model = make_model(current)
                optimizer = make_optimizer(model, current)
                row = train_epoch(model, optimizer, loader, current, 1, memory=False)
                row.update(workers=workers, run=repeat)
                rows.append(row)
                raw.append(row)
                mlflow.log_metrics(dict(throughput_samples_s=row['throughput_samples_s'],
                                        data_wait_fraction=row['data_wait_seconds'] / row['training_seconds']), step=repeat)
                print(f'workers={workers}, run {repeat}: {row["throughput_samples_s"]:.1f} samples/s', flush=True)
            summary.append(dict(workers=workers, runs=len(rows), warmups=1,
                                benchmark_samples=count, startup_seconds=startup,
                                median_samples_s=statistics.median(row['throughput_samples_s'] for row in rows),
                                median_data_wait_fraction=statistics.median(row['data_wait_seconds'] / row['training_seconds'] for row in rows),
                                run_id=run.info.run_id))
            artifact = RESULTS / 'worker_runs' / f'workers_{workers}.csv'
            save_csv(artifact, rows)
            mlflow.log_artifact(str(artifact))
        close_loader(loader)
    save_csv(RESULTS / 'worker_times_raw.csv', raw)
    save_csv(RESULTS / 'worker_summary.csv', summary)
    return max(summary, key=lambda row: row['median_samples_s'])['workers']
