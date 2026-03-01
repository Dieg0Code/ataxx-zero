from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path


def _should_skip(path: Path, skip_globs: list[str]) -> bool:
    posix = path.as_posix()
    return any(fnmatch.fnmatch(posix, pattern) for pattern in skip_globs)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _collect_violations(
    *,
    roots: list[Path],
    max_lines: int,
    skip_globs: list[str],
) -> list[tuple[str, int]]:
    violations: list[tuple[str, int]] = []
    for root in roots:
        if root.is_file():
            candidates = [root] if root.suffix == ".py" else []
        else:
            candidates = sorted(root.rglob("*.py"))
        for candidate in candidates:
            if _should_skip(candidate, skip_globs):
                continue
            line_count = _count_lines(candidate)
            rel = candidate.as_posix()
            if line_count > max_lines:
                violations.append((rel, line_count))
    return violations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce a maximum Python file length policy.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500,
        help="Maximum allowed lines per Python file.",
    )
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        default=[],
        help="Path to scan. Can be repeated. Defaults to src/tests/scripts/train.py.",
    )
    parser.add_argument(
        "--skip-glob",
        action="append",
        default=[".venv/**", "**/__pycache__/**", ".git/**"],
        help="Glob pattern to skip. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots_raw = args.paths or ["src", "tests", "scripts", "train.py"]
    roots: list[Path] = [Path(item) for item in roots_raw]

    violations = _collect_violations(
        roots=roots,
        max_lines=args.max_lines,
        skip_globs=list(args.skip_glob),
    )
    if violations:
        print(f"Python file length policy failed (> {args.max_lines} lines):")
        for rel, line_count in violations:
            print(f"  - {rel}: {line_count} lines")
        raise SystemExit(1)

    print(f"Python file length policy passed (max {args.max_lines} lines).")


if __name__ == "__main__":
    main()
