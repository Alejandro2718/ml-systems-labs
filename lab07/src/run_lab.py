import warnings

warnings.filterwarnings('ignore', message='The pynvml package is deprecated')

from print_versions import main as print_versions
from train_baseline import main as train_baseline
from prune_models import main as prune_models
from sparse_kernels import main as sparse_kernels
from distill import main as distill
from remeasure_latency import main as remeasure_latency
from summarize import main as summarize


if __name__ == '__main__':
    print_versions()
    train_baseline()
    prune_models()
    sparse_kernels()
    distill()
    remeasure_latency()
    summarize()
