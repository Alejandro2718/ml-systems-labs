import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

lines = [
    f"Python: {sys.version.split()[0]}",
    f"Platform: {platform.platform()}",
    "",
    "Installed packages:",
]

packages = sorted(
    (
        (dist.metadata.get("Name") or "Unknown", dist.version)
        for dist in distributions()
    ),
    key=lambda item: item[0].lower(),
)

lines.extend(f"{name}=={version}" for name, version in packages)
text = "\n".join(lines) + "\n"

output = Path(__file__).resolve().parents[1] / "results" / "versions.txt"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(text, encoding="utf-8")

print(text)
print(f"Version report saved to: {output}")
