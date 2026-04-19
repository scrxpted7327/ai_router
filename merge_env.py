#!/usr/bin/env python3
"""
Merge deploy/env.production into .env.

- Values from .env win when the same key exists in both files.
- Keys only in env.production are added (template defaults).
- Keys only in .env are appended at the bottom under a marker comment.
- Backs up the previous .env to .env.bak.<timestamp> before overwriting.

Usage:
  python merge_env.py
  python merge_env.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCTION = ROOT / "deploy" / "env.production"
DOTENV = ROOT / ".env"


def parse_env(path: Path) -> dict[str, str]:
    """KEY=value lines; ignores blanks and # comments; last wins per key."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k:
            out[k] = v
    return out


def merge_dicts(production: dict[str, str], existing: dict[str, str]) -> dict[str, str]:
    """Production first, then existing overrides."""
    merged = {**production, **existing}
    return merged


def build_output(
    production_lines: list[str],
    merged: dict[str, str],
    prod_keys: set[str],
) -> str:
    """Rewrite production file lines, substituting values from merged; append extras."""
    out_lines: list[str] = []
    key_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

    for line in production_lines:
        m = key_pattern.match(line.rstrip("\n"))
        if m:
            key, _tail = m.group(1), m.group(2)
            if key in merged:
                out_lines.append(f"{key}={merged[key]}")
                continue
        out_lines.append(line.rstrip("\n"))

    extra_keys = [k for k in merged if k not in prod_keys]
    if extra_keys:
        out_lines.append("")
        out_lines.append("# ── Additional keys from existing .env (not in env.production) ──")
        for k in sorted(extra_keys):
            out_lines.append(f"{k}={merged[k]}")

    return "\n".join(out_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge deploy/env.production with .env")
    ap.add_argument(
        "--production",
        type=Path,
        default=PRODUCTION,
        help="Template file (default: deploy/env.production)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DOTENV,
        help="Output path (default: .env)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print result to stdout; do not write files",
    )
    args = ap.parse_args()

    if not args.production.exists():
        raise SystemExit(f"Missing template: {args.production}")

    prod_text = args.production.read_text(encoding="utf-8")
    prod_lines = prod_text.splitlines(keepends=True)
    if not prod_lines:
        prod_lines = [""]

    production_map = parse_env(args.production)
    prod_keys = set(production_map.keys())
    existing_map = parse_env(args.out) if args.out.exists() else {}

    merged = merge_dicts(production_map, existing_map)
    body = build_output(
        [ln.rstrip("\r\n") for ln in prod_lines],
        merged,
        prod_keys,
    )

    if args.dry_run:
        print(body, end="")
        return

    if args.out.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = args.out.parent / f".env.bak.{stamp}"
        shutil.copy2(args.out, bak)
        print(f"Backed up existing .env -> {bak.name}")

    args.out.write_text(body, encoding="utf-8")
    print(f"Wrote {args.out} ({len(merged)} keys)")


if __name__ == "__main__":
    main()
