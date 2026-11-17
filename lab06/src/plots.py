import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=160)
    plt.close(fig)


def make_plots(config):
    plt.rcParams.update({'font.size':10, 'axes.grid':True, 'grid.alpha':0.25})
    workers = pd.read_csv(RESULTS / 'worker_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(workers.workers, workers.median_samples_s, 'o-')
    axes[0].set(xlabel='DataLoader workers', ylabel='Training samples/s', title='Median of five 100-batch passes')
    axes[1].plot(workers.workers, workers.median_data_wait_fraction * 100, 'o-')
    axes[1].set(xlabel='DataLoader workers', ylabel='Data wait (% of timed pass)', title='Parent-process waiting time')
    for ax in axes:
        ax.set_xticks(workers.workers)
    finish(fig, 'worker_study.png')
    batch = pd.read_csv(RESULTS / 'batch_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    series = [('median_throughput_samples_s', 'Training samples/s'),
              ('peak_tree_rss_mb', 'Peak process-tree RSS (MB)'),
              ('validation_accuracy', 'Final validation accuracy')]
    for ax, (column, label) in zip(axes, series):
        ax.plot(batch.batch_size, batch[column], 'o-')
        ax.set(xlabel='Physical batch size', ylabel=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(batch.batch_size, labels=batch.batch_size.astype(str))
    finish(fig, 'batch_study.png')
    schedules = pd.read_csv(RESULTS / 'schedule_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for index, optimizer in enumerate(config['studies']['optimizers']):
        subset = schedules.loc[schedules.optimizer.eq(optimizer)]
        for _, row in subset.iterrows():
            history = pd.read_csv(RESULTS / 'histories' / f'{row["name"]}.csv')
            axes[index, 0].plot(history.epoch, history.validation_accuracy, marker='.', label=row.scheduler)
            axes[index, 1].plot(history.epoch, history.learning_rate, marker='.', label=row.scheduler)
        axes[index, 0].set(title=optimizer, xlabel='Epoch', ylabel='Validation accuracy')
        axes[index, 1].set(title=optimizer, xlabel='Epoch', ylabel='Learning rate used')
        for ax in axes[index]:
            ax.legend(fontsize=8)
    finish(fig, 'schedule_comparison.png')
    for name, prefix, labels in [
        ('accumulation', ('batch256', 'accumulated'), ('Batch 256', 'Batch 64 x 4')),
        ('resume', ('uninterrupted', 'resumed'), ('Uninterrupted', 'Stopped after epoch 3, resumed'))]:
        frame = pd.read_csv(RESULTS / f'{name}_comparison.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            for index, (column, label) in enumerate(zip(prefix, labels)):
                ax.plot(frame.epoch, frame[f'{column}_validation_{metric}'],
                        marker='o' if index == 0 else 'x', linestyle='-' if index == 0 else '--', label=label)
            ax.set(xlabel='Epoch', ylabel=f'Validation {metric}')
            ax.set_xticks(frame.epoch)
            if name == 'resume':
                ax.axvline(config['studies']['interrupt_epoch'] + 0.5, color='grey', linestyle=':', label='Restart boundary')
            ax.legend(fontsize=8)
        finish(fig, f'{name}_comparison.png')
