"""
Curriculum-manifest handling for ``trainSB3.py --manifest``.

``prepare_curriculum.py`` writes a ``manifest.json`` describing every stage of
a curriculum chain (target file, run name, expected warm-start path). This
module resolves the stage for a given ``--n_qubits`` and fills in the
corresponding CLI arguments, so each stage can be launched as an independent
job (e.g. one SLURM job per stage).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

__all__ = ["resolve_manifest_args"]


def resolve_manifest_args(args: argparse.Namespace) -> bool:
    """Fill ``args`` in place from the manifest stage matching ``args.n_qubits``.

    Sets ``state="custom"``, ``custom_target_path``, ``warmstart_path``,
    ``run_name``, ``saving_path`` and ``prev_model_path`` (the previous stage's
    ``best_model.zip``, if recorded in its warm-start JSON and present on disk).

    Args:
        args (argparse.Namespace): parsed ``trainSB3.py`` arguments; read
            (``manifest``, ``n_qubits``, ``force``) and modified in place.

    Returns:
        bool: True if training should proceed; False if this stage should be
        skipped (``<run_name>_final.zip`` already exists and ``--no_force`` was
        given). Always True when no manifest is given.

    Raises:
        ValueError: if the manifest has no stage with ``n_qubits``.
        SystemExit: (exit code 1) if the previous stage's warm-start file does
            not exist yet.
    """
    if args.manifest is None:
        return True

    with open(args.manifest) as f:
        manifest = json.load(f)

    stage = next((s for s in manifest["stages"] if s["n_qubits"] == args.n_qubits), None)
    if stage is None:
        available = [s["n_qubits"] for s in manifest["stages"]]
        raise ValueError(f"No stage with n_qubits={args.n_qubits} in {args.manifest}. Available: {available}")

    args.state = "custom"
    args.custom_target_path = stage["target_path"]
    args.warmstart_path = stage["warmstart_path"]
    args.run_name = stage["run_name"]
    args.saving_path = manifest["out_dir"]

    os.makedirs(args.saving_path, exist_ok=True)
    final_path = os.path.join(args.saving_path, f"{args.run_name}_final.zip")
    if os.path.exists(final_path) and not args.force:
        print(f"{final_path} already exists -- stage {args.n_qubits} looks done, skipping "
              f"(pass --force to retrain).")
        return False

    if args.warmstart_path is not None and not os.path.exists(args.warmstart_path):
        print(f"ERROR: stage {args.n_qubits} needs {args.warmstart_path} (produced by the "
              f"PREVIOUS stage), but it doesn't exist yet.", file=sys.stderr)
        print("The previous stage's job hasn't finished, failed, or ran out of wall-clock "
              "time before solving its target. Run/re-run it first.", file=sys.stderr)
        sys.exit(1)

    # Network warm-start: the previous stage's warmstart JSON may carry a
    # "model_path" pointing at its best_model.zip.
    args.prev_model_path = None
    if args.warmstart_path is not None:
        with open(args.warmstart_path) as f:
            prev_ws = json.load(f)
        candidate = prev_ws.get("model_path")
        if candidate and os.path.exists(candidate):
            args.prev_model_path = candidate

    print(f"[manifest] stage {args.n_qubits} qubits -> run_name={args.run_name}, "
          f"target={args.custom_target_path}, warmstart={args.warmstart_path}, "
          f"prev_model={args.prev_model_path}, saving_path={args.saving_path}")
    return True
