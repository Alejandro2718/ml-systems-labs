import platform
from importlib.metadata import distributions, version
from pathlib import Path

import torch
import torchvision


def main():
    names = {item.metadata['Name'] for item in distributions() if item.metadata.get('Name')}
    rows = sorted((name, version(name)) for name in names)
    text = f'Python: {platform.python_version()}\nPlatform: {platform.platform()}\n'
    text += f'PyTorch runtime: {torch.__version__}\nTorchvision runtime: {torchvision.__version__}\n\n'
    text += '\n'.join(f'{name}=={version}' for name,version in rows) + '\n'
    path = Path(__file__).resolve().parents[1] / 'results/versions.txt'
    path.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
