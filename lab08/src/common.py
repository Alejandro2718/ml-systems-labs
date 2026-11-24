import importlib.util
import io
import platform
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
RESULTS = LAB / 'results'
DATA = ROOT / 'data' / 'lab08'
LAB07 = ROOT / 'lab07'
BACKEND = 'fbgemm'
CALIBRATION_IMAGES = 500


def lab07_module(name):
    key = f'lab07_{name}'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, LAB07 / 'src' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


shared = lab07_module('common')
models = lab07_module('models')
SEED = shared.SEED
save_csv = shared.save_csv
evaluate = shared.evaluate
eval_loader = shared.eval_loader
latency_row = shared.latency_row
load_data = shared.load_data


def setup():
    shared.setup()
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings('ignore', message='quantized_resize_cpu_')
    if platform.machine().lower() in ('arm64', 'aarch64'):
        raise RuntimeError('Use qnnpack on ARM; these measurements were planned for x86 fbgemm')
    if BACKEND not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f'{BACKEND} is not available')
    torch.backends.quantized.engine = BACKEND
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def lab07_choices():
    summary = pd.read_csv(LAB07 / 'results' / 'efficiency_summary.csv')
    pruned = summary.loc[summary.variant.str.startswith('filters_'), 'variant'].iloc[0]
    return 'baseline', pruned, 'student_distilled'


def load_lab07(name):
    if name.startswith('filters_'):
        shape, _ = models.filter_pruned_teacher(models.teacher_cnn(), int(name.split('_')[1]) / 100)
        return shared.load_state(shape, name)
    if name.startswith('student'):
        return shared.load_state(models.student_cnn(), name)
    return shared.load_state(models.teacher_cnn(), name)


def state_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())
