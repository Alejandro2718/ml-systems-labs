import platform
from importlib.metadata import distributions, version, PackageNotFoundError
from pathlib import Path
from packaging.version import Version

root = Path(__file__).resolve().parents[2]
output = root / "lab01" / "results" / "versions.txt"
output.parent.mkdir(parents=True, exist_ok=True)

lines = [
    f"Python: {platform.python_version()}",
    f"Platform: {platform.platform()}",
    "",
    "Required package verification:",
]

requirements = root / "requirements.txt"

for line in requirements.read_text(encoding="utf-8-sig").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue

    name, expected = line.split("==", 1)

    try:
        installed = version(name)
        status = "OK" if Version(installed) == Version(expected) else "VERSION MISMATCH"
        lines.append(
            f"{name}: required={expected}, installed={installed} [{status}]"
        )
    except PackageNotFoundError:
        lines.append(f"{name}: required={expected} [NOT INSTALLED]")

lines.extend(["", "All installed packages:"])

packages = sorted(
    (dist.metadata["Name"], dist.version)
    for dist in distributions()
    if dist.metadata["Name"]
)

lines.extend(f"{name}=={installed}" for name, installed in packages)

text = "\n".join(lines) + "\n"
output.write_text(text, encoding="utf-8")
print(text)
print(f"Version report saved to: {output}")

