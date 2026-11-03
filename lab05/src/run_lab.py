from print_versions import main as print_versions
from verify_gradients import main as verify_gradients
from train_models import main as train_models
from measure_framework import main as measure_framework


if __name__ == '__main__':
    print_versions()
    verify_gradients()
    train_models()
    measure_framework()
