#!/usr/bin/env python3
"""
Backfill train/epoch, train/iteration, val/epoch, test/epoch on existing W&B runs.

Older runs logged metrics without these keys. W&B merges new keys into existing
steps when you resume the same run id and log again with the same step numbers.

Requires: pip install wandb
Usage (single run):
  wandb login
  python scripts/backfill_wandb_epoch_metrics.py \\
    --entity YOUR_ENTITY --project YOUR_PROJECT --run-id RUN_ID \\
    --train-batches-per-epoch 469

Usage (all runs in project):
  python scripts/backfill_wandb_epoch_metrics.py \\
    --entity YOUR_ENTITY --project YOUR_PROJECT --all \\
    --train-batches-per-epoch 469

--train-batches-per-epoch is optional; if omitted, train/iteration is skipped.
With --all, the same value is used for every run (typical when batch size is shared).

Live backfill calls ``wandb.init(resume=...)`` and needs the local wandb-core service.
If you see ``ServicePollForTokenError`` on Windows, try: WSL/Linux, ``pip install -U wandb``,
or set ``WANDB__SERVICE_WAIT`` higher; ``--dry-run`` only uses the API and always works.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Tuple

# See wandb.sdk.wandb_settings.Settings.update_from_env_vars: WANDB__SERVICE_WAIT -> x_service_wait
_DEFAULT_SERVICE_WAIT = "300"


def _row_has_mode_metrics(row: Any, mode: str) -> bool:
    import math

    p = f"{mode}/"
    skip = {f"{mode}/epoch", f"{mode}/iteration"}
    for c in row.index:
        if not str(c).startswith(p) or c in skip:
            continue
        v = row[c]
        if v is None:
            continue
        try:
            if isinstance(v, float) and (math.isnan(v)):
                continue
        except TypeError:
            pass
        return True
    return False


def backfill_one_run(
    wandb_mod: Any,
    api: Any,
    entity: str,
    project: str,
    run_id: str,
    train_batches_per_epoch: Optional[int],
    dry_run: bool,
) -> Tuple[int, str]:
    """Returns (exit_code, message). exit_code 0 = ok, 1 = skip/fail."""
    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
    except Exception as e:
        return 1, f"{path}: could not load run ({e})"

    df = run.history(samples=10**9)
    if df.empty:
        return 1, f"{path}: no history, skip"

    if "_step" not in df.columns:
        return 1, f"{path}: no _step column, skip"

    steps = sorted({int(round(s)) for s in df["_step"].dropna().unique()})

    def build_log_for_step(step: int) -> dict:
        row = df[df["_step"] == step].iloc[-1]
        log: dict = {}
        for mode in ("train", "val", "test"):
            if _row_has_mode_metrics(row, mode):
                log[f"{mode}/epoch"] = float(step + 1)
        if (
            train_batches_per_epoch is not None
            and train_batches_per_epoch > 0
            and _row_has_mode_metrics(row, "train")
        ):
            log["train/iteration"] = float(
                (step + 1) * train_batches_per_epoch - 1
            )
        return log

    if dry_run:
        sample = [build_log_for_step(s) for s in steps[:3]]
        extra = f", ... +{len(steps) - 3} steps" if len(steps) > 3 else ""
        return 0, f"{path}: would backfill {len(steps)} steps (sample: {sample}{extra})"

    # Env is reliable across subprocesses; default 30s often fails on Windows.
    os.environ.setdefault("WANDB__SERVICE_WAIT", _DEFAULT_SERVICE_WAIT)
    wandb_mod.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
    )
    try:
        for step in steps:
            log = build_log_for_step(step)
            if log:
                wandb_mod.log(log, step=int(step))
    finally:
        wandb_mod.finish()

    return 0, f"{path}: backfilled {len(steps)} steps"


def main() -> int:
    try:
        import wandb
    except ImportError as e:
        print("Install dependencies: pip install wandb", file=sys.stderr)
        raise SystemExit(1) from e

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument(
        "--run-id",
        default=None,
        help="Single W&B run id (short id from UI). Omit when using --all.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Backfill every run in the project (entity + project).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="With --all, process at most this many runs (order is API default, usually newest first).",
    )
    ap.add_argument(
        "--train-batches-per-epoch",
        type=int,
        default=None,
        help="If set, writes train/iteration = (step+1)*B - 1 (same formula as LibMTL trainer).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without calling wandb.init / wandb.log",
    )
    args = ap.parse_args()

    if args.all and args.run_id:
        print("Use either --all or --run-id, not both.", file=sys.stderr)
        return 1
    if not args.all and not args.run_id:
        print("Specify --run-id for one run or --all for every run in the project.", file=sys.stderr)
        return 1

    os.environ.setdefault("WANDB__SERVICE_WAIT", _DEFAULT_SERVICE_WAIT)

    api = wandb.Api()

    if not args.all:
        code, msg = backfill_one_run(
            wandb,
            api,
            args.entity,
            args.project,
            args.run_id,
            args.train_batches_per_epoch,
            args.dry_run,
        )
        print(msg)
        return code

    runs = list(api.runs(f"{args.entity}/{args.project}"))
    if args.limit is not None:
        runs = runs[: args.limit]

    failed: list[str] = []
    for i, run in enumerate(runs):
        rid = run.id
        print(f"[{i + 1}/{len(runs)}] {rid}", flush=True)
        code, msg = backfill_one_run(
            wandb,
            api,
            args.entity,
            args.project,
            rid,
            args.train_batches_per_epoch,
            args.dry_run,
        )
        print(f"  {msg}")
        if code != 0:
            failed.append(rid)

    if failed:
        print(f"Skipped or failed: {len(failed)} run(s)", file=sys.stderr)
        return 1
    print(f"Done: processed {len(runs)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
