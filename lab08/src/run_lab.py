import warnings

warnings.filterwarnings('ignore', message='The pynvml package is deprecated')

from print_versions import main as print_versions
from manual_quantization import main as manual_quantization
from quantize_models import main as quantize_models
from export_onnx import main as export_onnx
from plots import main as plots


if __name__ == '__main__':
    print_versions()
    manual_quantization()
    quantize_models()
    export_onnx()
    plots()
