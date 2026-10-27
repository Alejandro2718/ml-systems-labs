import csv
import random
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / "results"
DATA = LAB.parent / "data"
CHECKPOINTS = DATA / "lab03_models"
SEED = 42
THREADS = 4


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
