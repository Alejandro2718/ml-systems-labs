import copy
import csv
import gzip
import io
import random
from pathlib import Path
from statistics import median
from time import perf_counter, perf_counter_ns

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
RESULTS = LAB / 'results'
DATA = ROOT / 'data'
CHECKPOINTS = DATA / 'lab07'
SEED = 42
THREADS = 4
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
BASELINE_EPOCHS = 10
STUDENT_EPOCHS = 10
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 3e-4
SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
FILTER_RATIOS = [0.1, 0.25, 0.5]
TEMPERATURE = 4.0
ALPHA = 0.5


def setup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(THREADS)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST(DATA, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=transform)
    assert len(train) == 60000 and len(test) == 10000

    def cache(dataset):
        batches = list(DataLoader(dataset, batch_size=1000, shuffle=False))
        return TensorDataset(torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches]))

    train, test = cache(train), cache(test)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(SEED))
    return Subset(train, order[:55000].tolist()), Subset(train, order[55000:].tolist()), test


def train_loader(dataset, seed=SEED):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))


def eval_loader(dataset):
    return DataLoader(dataset, batch_size=512)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_sum = correct = count = 0
    for images, labels in loader:
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return loss_sum / count, correct / count


def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1), reduction='batchmean')
    return alpha * hard + (1 - alpha) * temperature ** 2 * soft


def train(model, data, epochs, lr, label, validation=None, teacher=None, seed=SEED):
    loader = train_loader(data, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    if teacher is not None:
        teacher.eval()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = seen = 0
        start = perf_counter()
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if teacher is None:
                loss = F.cross_entropy(logits, labels)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = distillation_loss(logits, teacher_logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        seconds = perf_counter() - start
        row = dict(run=label, epoch=epoch, train_loss=loss_sum / seen, epoch_seconds=seconds)
        if validation is not None:
            row['validation_loss'], row['validation_accuracy'] = evaluate(model, eval_loader(validation))
        history.append(row)
        print(f"{label} {epoch}/{epochs} loss={row['train_loss']:.4f} "
              f"val={row.get('validation_accuracy', float('nan')):.4f} ({seconds:.1f}s)", flush=True)
    return history


@torch.inference_mode()
def latency(model, batch, runs, warmups=5):
    model.eval()
    x = torch.randn(batch, 1, 28, 28, generator=torch.Generator().manual_seed(SEED))
    for _ in range(warmups):
        model(x)
    measured = []
    for _ in range(runs):
        start = perf_counter_ns()
        model(x)
        measured.append((perf_counter_ns() - start) / 1e6)
    return median(measured), float(np.percentile(measured, 95))


def latency_row(model):
    b1, b1_p95 = latency(model, 1, 300)
    b128, b128_p95 = latency(model, 128, 40)
    return dict(batch1_median_ms=b1, batch1_p95_ms=b1_p95, batch128_median_ms=b128, batch128_p95_ms=b128_p95)


def serialized_bytes(state, compress=False):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    data = buffer.getvalue()
    return len(gzip.compress(data, compresslevel=9, mtime=0)) if compress else len(data)


def sparse_state(model):
    state = {}
    for name, value in model.state_dict().items():
        state[name] = value.to_sparse_csr() if value.dim() == 2 else (
            value.flatten(1).to_sparse_csr() if value.dim() == 4 else value)
    return state


def size_row(model):
    dense = model.state_dict()
    return dict(dense_bytes=serialized_bytes(dense), dense_gzip_bytes=serialized_bytes(dense, compress=True),
                sparse_csr_bytes=serialized_bytes(sparse_state(model)))


def save_model(model, name):
    torch.save(model.state_dict(), CHECKPOINTS / f'{name}.pth')


def load_state(model, name):
    model.load_state_dict(torch.load(CHECKPOINTS / f'{name}.pth', weights_only=True))
    return model.eval()


def clone(model):
    return copy.deepcopy(model)


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
