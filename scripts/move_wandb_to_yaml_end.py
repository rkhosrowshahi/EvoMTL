"""One-off: move # Weights & Biases + wandb_* keys to end of each multimnist config YAML."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "examples" / "multimnist" / "configs"

HEADER_RE = re.compile(r"^\s*#\s*Weights.*Biases\s*$")


def find_wandb_slice(lines: list[str]) -> tuple[int, int, list[str]] | None:
    """Return (start, end_exclusive, block_lines) or None. block_lines include header + wandb_*."""
    n = len(lines)
    header_i: int | None = None
    first_wandb: int | None = None
    for i, line in enumerate(lines):
        if HEADER_RE.match(line):
            header_i = i
            break
        if line.lstrip().startswith("wandb_"):
            first_wandb = i
            break
    if header_i is not None:
        i0 = header_i
        j = header_i + 1
        while j < n and lines[j].lstrip().startswith("wandb_"):
            j += 1
        block = lines[i0:j]
    elif first_wandb is not None:
        i0 = first_wandb
        j = first_wandb
        while j < n and lines[j].lstrip().startswith("wandb_"):
            j += 1
        block = ["# Weights & Biases"] + lines[i0:j]
    else:
        return None
    if not block:
        return None
    start = i0
    if start > 0 and lines[start - 1].strip() == "":
        start -= 1
    return start, j, block


def transform(text: str) -> tuple[str, bool]:
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    orig_norm = "\n".join(lines) + "\n"
    sl = find_wandb_slice(lines)
    if sl is None:
        return text, False
    start, end_excl, block = sl
    rest = lines[:start] + lines[end_excl:]
    while rest and rest[-1].strip() == "":
        rest.pop()
    out_lines = rest + [""] + block
    new_text = "\n".join(out_lines) + "\n"
    return new_text, new_text != orig_norm


def main() -> None:
    changed = 0
    for path in sorted(ROOT.rglob("*.yaml")):
        old = path.read_text(encoding="utf-8")
        new, did = transform(old)
        if did:
            path.write_text(new, encoding="utf-8", newline="\n")
            changed += 1
    print(f"Updated {changed} files under {ROOT}")


if __name__ == "__main__":
    main()
