import random
from statistics import median

from common import RESULTS, SEED, latency, save_csv, setup
from summarize import build

NAMES = (['baseline'] + [f'{m}_{s}' for m in ('global', 'layerwise') for s in (0, 25, 50, 75, 90, 95)
                         if f'{m}_{s}' != 'layerwise_0']
         + ['filters_10', 'filters_25', 'filters_50', 'student_plain', 'student_distilled'])
ROUNDS = 5


def main():
    setup()
    models = {name: build(name) for name in NAMES}
    order = list(NAMES)
    shuffle = random.Random(SEED)
    samples = {name: {1: [], 128: []} for name in NAMES}
    for round_index in range(ROUNDS):
        shuffle.shuffle(order)
        for name in order:
            samples[name][1].append(latency(models[name], 1, 100)[0])
            samples[name][128].append(latency(models[name], 128, 12)[0])
        print(f'latency round {round_index + 1}/{ROUNDS} done', flush=True)
    rows = []
    for name in NAMES:
        row = dict(run=name, rounds=ROUNDS)
        for batch in (1, 128):
            values = samples[name][batch]
            row.update({f'batch{batch}_median_ms': median(values), f'batch{batch}_round_min_ms': min(values),
                        f'batch{batch}_round_max_ms': max(values)})
        rows.append(row)
        print(f"{name}: batch 1 {row['batch1_median_ms']:.3f} ms, batch 128 {row['batch128_median_ms']:.2f} ms",
              flush=True)
    save_csv(RESULTS / 'latency_interleaved.csv', rows)


if __name__ == '__main__':
    main()
