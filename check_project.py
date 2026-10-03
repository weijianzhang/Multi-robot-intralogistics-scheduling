"""Statically validate release sources without importing or running algorithms."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {".venv", "venv", "__pycache__", ".git"}
ROOT_SOURCES = ("VNS.py", "VNS-HE.py", "Metaheuristics.py", "Gurobi.py", "lower_bound.py")


def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*.py")
        if not EXCLUDED.intersection(path.relative_to(ROOT).parts)
    )
    errors = []
    local_modules = {path.stem for path in files}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"Syntax: {path.relative_to(ROOT)}: {exc}")
            continue
        # Check sibling/root imports without loading third-party dependencies.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in local_modules:
                sibling = path.parent / f"{node.module}.py"
                at_root = ROOT / f"{node.module}.py"
                if not sibling.is_file() and not at_root.is_file():
                    errors.append(f"Missing local import: {path.relative_to(ROOT)} -> {node.module}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.endswith(".py") and ("/" in node.value or node.value in ROOT_SOURCES):
                    target = ROOT / node.value
                    if not target.is_file():
                        errors.append(f"Missing source reference: {path.relative_to(ROOT)} -> {node.value}")
    for name in ROOT_SOURCES:
        if not (ROOT / name).is_file():
            errors.append(f"Missing core source: {name}")
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(f"Syntax and local source references OK: {len(files)} Python files")
    print("No algorithms, experiments, or plotting scripts were executed.")


if __name__ == "__main__":
    main()
