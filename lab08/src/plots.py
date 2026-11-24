import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS


def main():
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25})
    frame = pd.read_csv(RESULTS / 'variant_comparison.csv')
    colors = {'fp32': 'tab:blue', 'dynamic_int8_linear': 'tab:orange', 'static_int8': 'tab:green'}
    markers = {'baseline': 'o', 'pruned': 's', 'distilled': '^'}
    frame['family'] = frame.source_model.map(lambda n: 'pruned' if n.startswith('filters_') else
                                             'distilled' if n.startswith('student') else 'baseline')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    columns = [('batch1_median_ms', 'Batch-1 median latency (ms)'),
               ('batch128_median_ms', 'Batch-128 median latency (ms)'),
               ('state_dict_bytes', 'Serialized state_dict size (bytes)')]
    for ax, (column, label) in zip(axes, columns):
        for _, row in frame.iterrows():
            ax.scatter(row[column], row.test_accuracy, marker=markers[row.family], color=colors[row.method], s=60)
        ax.set(xlabel=label, ylabel='Test accuracy', xscale='log')
    handles = [plt.Line2D([], [], color=c, marker='o', linestyle='', label=m) for m, c in colors.items()]
    handles += [plt.Line2D([], [], color='grey', marker=k, linestyle='', label=f'{f} model') for f, k in markers.items()]
    axes[0].legend(handles=handles, fontsize=8, loc='center left')
    fig.suptitle('Fashion-MNIST variants on x86 CPU, fbgemm backend, 4 threads')
    fig.tight_layout()
    fig.savefig(RESULTS / 'variant_tradeoffs.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
