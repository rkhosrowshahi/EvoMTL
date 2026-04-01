"""Run every YAML config in a folder sequentially.

Usage
-----
    python runner.py --dir configs/mnist/cagrad/
    python runner.py --dir configs/mnist/cagrad/ --gpu_id 1
    python runner.py --dir configs/fashion/famo/ --dry_run
    python runner.py --dir configs/mnist/cagrad/ --from my_run
    python runner.py --dir configs/mnist/cagrad/ --from my_run.yaml
    python runner.py --dir configs/mnist/cagrad/ --from_index 3

Any extra flags are forwarded verbatim to ``main.py`` and will override the
corresponding YAML values.

All ``*.yaml`` and ``*.yml`` files in the directory are processed in
**alphabetical** order by filename.  Use ``--from`` to skip configs before a
given file, or ``--from_index N`` (1-based) to start at the Nth config in that
order.  A summary table is printed at the end.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def slice_from(yamls: list, start: str) -> list:
    """Return configs from ``start`` onward (inclusive). ``start`` is a stem or filename."""
    start = start.strip()
    if not start.endswith(('.yaml', '.yml')):
        start_name = f'{start}.yaml'
    else:
        start_name = start
    want_stem = Path(start_name).stem
    for i, p in enumerate(yamls):
        if p.name == start_name or p.stem == want_stem:
            return yamls[i:]
    sys.exit(
        f'[runner] ERROR: no config matching --from {start!r} '
        f'(expected one of: {", ".join(p.name for p in yamls)})'
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run all YAML configs in a folder sequentially via main.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dir', required=True, type=str,
                        help='directory containing *.yaml / *.yml configs')
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--from', dest='start_from', default=None, metavar='NAME',
                       help='resume: skip configs before this file (stem or full name, e.g. run_b or run_b.yaml)')
    start.add_argument('--from_index', dest='start_from_index', type=int, default=None,
                       metavar='N',
                       help='resume: 1-based index in sorted order (e.g. 3 = third YAML)')
    parser.add_argument('--dry_run', action='store_true',
                        help='print commands without executing them')
    # Everything else is forwarded to main.py
    args, extra = parser.parse_known_args()
    return args, extra


def main():
    args, extra_flags = parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        sys.exit(f'[runner] ERROR: "{folder}" is not a directory.')

    yamls = sorted(
        list(folder.glob('*.yaml')) + list(folder.glob('*.yml')),
        key=lambda p: p.name,
    )
    if not yamls:
        sys.exit(f'[runner] ERROR: no *.yaml or *.yml files found in "{folder}".')

    total_in_folder = len(yamls)
    if args.start_from_index is not None:
        n = args.start_from_index
        if n < 1 or n > total_in_folder:
            sys.exit(
                f'[runner] ERROR: --from_index {n} out of range '
                f'(use 1..{total_in_folder})'
            )
        yamls = yamls[n - 1:]
        print(f'[runner] Found {total_in_folder} config(s) in "{folder}"; '
              f'running {len(yamls)} starting at index {n} ({yamls[0].name})')
    elif args.start_from:
        yamls = slice_from(yamls, args.start_from)
        print(f'[runner] Found {total_in_folder} config(s) in "{folder}"; '
              f'running {len(yamls)} starting from {yamls[0].name}')
    else:
        print(f'[runner] Found {len(yamls)} config(s) in "{folder}"')
    if extra_flags:
        print(f'[runner] Extra flags forwarded to main.py: {extra_flags}')
    print()

    results = []

    for idx, yaml_path in enumerate(yamls, 1):
        cmd = [sys.executable, 'main.py', '--config', str(yaml_path)] + extra_flags
        print(f'[{idx}/{len(yamls)}] Running: {" ".join(cmd)}')

        if args.dry_run:
            results.append((yaml_path.name, 'DRY-RUN', 0.0))
            continue

        t0 = time.time()
        ret = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = 'OK' if ret.returncode == 0 else f'FAILED (exit {ret.returncode})'
        results.append((yaml_path.name, status, elapsed))
        print(f'    → {status}  ({elapsed:.1f}s)\n')

    # ── summary ──────────────────────────────────────────────────────────────
    print('=' * 55)
    print(f'{"Config":<30}  {"Status":<18}  {"Time":>6}')
    print('-' * 55)
    for name, status, elapsed in results:
        time_str = f'{elapsed:.1f}s' if elapsed else '-'
        print(f'{name:<30}  {status:<18}  {time_str:>6}')
    print('=' * 55)

    n_ok = sum(1 for _, s, _ in results if s == 'OK')
    n_fail = len(results) - n_ok
    print(f'{n_ok}/{len(results)} succeeded', end='')
    if n_fail:
        print(f'  ({n_fail} FAILED)', end='')
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
