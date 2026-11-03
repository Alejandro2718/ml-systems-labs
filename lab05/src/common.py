import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / 'results'
DATA = LAB.parent / 'data' / 'lab05'
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.1


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def load_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=SEED)
    return train_test_split(X.astype(np.float64), y.astype(np.float64),
                            test_size=0.3, stratify=y, random_state=SEED)


def initial_weights():
    rng = np.random.default_rng(SEED)
    return {'w1': rng.normal(0, 0.5, (8, 2)), 'b1': np.zeros(8),
            'w2': rng.normal(0, 0.5, (1, 8)), 'b2': np.zeros(1)}


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
