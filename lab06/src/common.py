import copy
import csv
import hashlib
import importlib.util
import random
import threading
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
RESULTS = LAB / 'results'
DATA = ROOT / 'data' / 'lab06'


def load_config(path=None):
    path = Path(path) if path else LAB / 'config.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if config['device'] != 'cpu':
        raise ValueError('These measurements use the CPU environment from Lab 3')
    return config


def setup(config):
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config['threads'])
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def changed(config, **settings):
    result = copy.deepcopy(config)
    result['training'].update(settings)
    return result


def config_hash(config):
    text = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def make_model(config):
    with (ROOT / 'lab03/results/training_summary.csv').open(newline='') as file:
        best = max(csv.DictReader(file), key=lambda row: float(row['test_accuracy']))
    if best['model'] != config['model']:
        raise ValueError('The configured model must be the best recorded Lab 3 model')
    path = ROOT / 'lab03/src/models.py'
    spec = importlib.util.spec_from_file_location('lab03_models', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_model(config['model'])


def make_datasets(config):
    values = config['dataset']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (values['normalize_mean'],), (values['normalize_std'],))])
    train = datasets.FashionMNIST(ROOT / 'data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(ROOT / 'data', train=False, download=True, transform=transform)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed']))
    split = values['train_samples']
    if len(train) != split + values['validation_samples'] or len(test) != values['test_samples']:
        raise ValueError('Unexpected Fashion-MNIST sizes')
    return Subset(train, order[:split].tolist()), Subset(train, order[split:].tolist()), test


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, config, training=True):
    values = config['training']
    workers = values['num_workers'] if training else 0
    sampler = EpochSampler(dataset, config['seed']) if training else None
    options = dict(batch_size=values['batch_size'], sampler=sampler, num_workers=workers,
                   worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config['seed']),
                   persistent_workers=workers > 0, drop_last=False)
    if workers:
        options['prefetch_factor'] = 2
    return DataLoader(dataset, **options)


def close_loader(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()


def make_optimizer(model, config):
    values = config['training']
    kwargs = dict(lr=values['learning_rate'], weight_decay=values['weight_decay'])
    if values['optimizer'] == 'SGD':
        return torch.optim.SGD(model.parameters(), momentum=values['momentum'], **kwargs)
    if values['optimizer'] == 'Adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError('Unknown optimizer')


def make_scheduler(optimizer, config):
    values = config['training']
    if values['scheduler'] == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=values['step_size'], gamma=values['gamma'])
    if values['scheduler'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=values['epochs'], eta_min=values['cosine_min_lr'])
    if values['scheduler'] == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=values['gamma'], patience=values['plateau_patience'])
    raise ValueError('Unknown scheduler')


class MemoryMonitor:
    def __init__(self, interval):
        self.interval = interval
        self.stop = threading.Event()
        self.parent_peak = 0
        self.tree_peak = 0

    def sample(self):
        process = psutil.Process()
        parent = process.memory_info().rss
        total = parent
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.parent_peak = max(self.parent_peak, parent)
        self.tree_peak = max(self.tree_peak, total)

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        return self

    def watch(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __exit__(self, *args):
        self.sample()
        self.stop.set()
        self.thread.join()


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
