import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from common import DATA, LAB, RESULTS, ROOT, changed, load_config, save_csv


def command(task, config, name=None, study=None, stop=None, resume=None):
    path = DATA / 'configs' / f'{name or task}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    args = [sys.executable, '-B', str(LAB / 'src/train.py'), '--config', str(path), '--task', task]
    if name:
        args += ['--name', name, '--study', study]
    if stop:
        args += ['--stop-after', str(stop)]
    if resume:
        args += ['--resume', str(resume)]
    subprocess.run(args, cwd=ROOT, check=True)
    if name:
        return json.loads((RESULTS / 'runs' / f'{name}.json').read_text())


def summary_row(config, result):
    history = pd.read_csv(RESULTS / 'histories' / f'{result["name"]}.csv')
    values = config['training']
    return dict(name=result['name'], run_id=result['run_id'], batch_size=values['batch_size'],
                accumulation_steps=values['accumulation_steps'], num_workers=values['num_workers'],
                optimizer=values['optimizer'], scheduler=values['scheduler'], learning_rate=values['learning_rate'],
                epochs=values['epochs'], median_throughput_samples_s=float(history.throughput_samples_s.median()),
                peak_parent_rss_mb=float(history.peak_parent_rss_mb.max()),
                peak_tree_rss_mb=float(history.peak_tree_rss_mb.max()),
                validation_loss=result['validation_loss'], validation_accuracy=result['validation_accuracy'],
                test_loss=result['test_loss'], test_accuracy=result['test_accuracy'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reuse-workers', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--restore-tracking', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.restore_tracking:
        from tracking import restore_tracking
        restore_tracking()
        return
    if not args.finalize_only:
        if not args.reuse_workers:
            command('workers', config)
        workers = pd.read_csv(RESULTS / 'worker_summary.csv')
        selected = int(workers.loc[workers.median_samples_s.idxmax(), 'workers'])
        config = changed(config, num_workers=selected)
        (RESULTS / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        batch_rows = []
        for batch in config['studies']['batch_sizes']:
            current = changed(config, batch_size=batch, epochs=config['studies']['batch_epochs'])
            result = command('train', current, f'batch_{batch}', 'batch_size')
            batch_rows.append(summary_row(current, result))
            save_csv(RESULTS / 'batch_summary.csv', batch_rows)
        schedule_rows = []
        for optimizer, learning_rate in config['studies']['optimizers'].items():
            for scheduler in config['studies']['schedulers']:
                current = changed(config, epochs=config['studies']['schedule_epochs'],
                                  optimizer=optimizer, learning_rate=learning_rate, scheduler=scheduler)
                name = f'{optimizer.lower()}_{scheduler.lower()}'
                result = command('train', current, name, 'optimizer_schedule')
                schedule_rows.append(summary_row(current, result))
                save_csv(RESULTS / 'schedule_summary.csv', schedule_rows)
        command('profile', config)
        current = changed(config, batch_size=config['studies']['accumulation_batch'],
                          accumulation_steps=config['studies']['accumulation_steps'])
        result = command('train', current, 'accumulation_64x4', 'gradient_accumulation')
        save_csv(RESULTS / 'accumulation_summary.csv', [summary_row(current, result)])
        command('train', config, 'resume_check', 'resume', stop=config['studies']['interrupt_epoch'])
        command('train', config, 'resume_check', 'resume', resume=DATA / 'checkpoints/resume_check.pth')
    else:
        config = load_config(RESULTS / 'resolved_config.yaml')
    from checks import check_accumulation, check_resume
    check_accumulation(config)
    check_resume(config)
    from reporting import write_reports
    from plots import make_plots
    from tracking import export_tracking
    from print_versions import main as print_versions
    print_versions()
    make_plots(config)
    write_reports(config)
    export_tracking(config)
    print('Lab 6 pipeline and evidence completed.', flush=True)


if __name__ == '__main__':
    main()
