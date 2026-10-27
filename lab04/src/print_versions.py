import platform
from importlib.metadata import distributions
from pathlib import Path


def main():
    versions = sorted((dist.metadata.get('Name', 'Unknown'), dist.version) for dist in distributions())
    text = f'Python: {platform.python_version()}\nPlatform: {platform.platform()}\n\n'
    text += '\n'.join(f'{name}=={version}' for name, version in versions) + '\n'
    path = Path(__file__).resolve().parents[1] / 'results' / 'versions.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
