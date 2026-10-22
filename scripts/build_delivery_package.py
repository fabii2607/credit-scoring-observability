"""Empacota fontes/configuração/docs públicas sem dados, segredos ou modelos."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "tests", "scripts", "monitoring", "docs", "notebooks", ".github"}
FILES = {
    "README.md",
    "Makefile",
    "params.yaml",
    "pyproject.toml",
    "uv.lock",
    "docker-compose.yml",
    ".python-version",
    ".env.example",
    ".gitignore",
    ".pre-commit-config.yaml",
}


def build():
    raw = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
    )
    names = sorted({s.decode() for s in raw.split(b"\0") if s})
    output = ROOT / "reports/release/pacote-2.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            relative = Path(name)
            if relative.parts[0] not in DIRECTORIES and name not in FILES:
                continue
            source = ROOT / relative
            if (
                not source.is_file()
                or source.is_symlink()
                or "__pycache__" in relative.parts
            ):
                continue
            archive.write(source, arcname=Path("pacote-2") / relative)
            count += 1
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n")
    print(f"Pacote local criado: {output.relative_to(ROOT)}, {count} arquivos.")


if __name__ == "__main__":
    build()
