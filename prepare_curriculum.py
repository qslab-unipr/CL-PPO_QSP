"""
prepare_curriculum.py -- generate a whole curriculum chain and its manifest.

Run this ONCE before submitting any training job. It writes the target of
every stage (``target_<k>.npy`` for k = start_qubits .. n_qubits) and a
``manifest.json`` that ``trainSB3.py --manifest`` uses to configure each stage::

    python prepare_curriculum.py --n_qubits 10 --state_type random --seed 0
    python prepare_curriculum.py --n_qubits 10 --state_type dicke
    python prepare_curriculum.py --n_qubits 10 --state_type W

How the stage targets are built:

* ``random``: a genuinely random n-qubit target is sampled, then the dominant
  Schmidt vector across the last qubit is extracted recursively
  (n -> n-1 -> ... -> start_qubits). Each stage solves the dominant
  component of the next one. ``--seed`` makes it reproducible (if omitted, a
  seed is drawn from OS entropy and recorded in the manifest).
* ``dicke`` (k=2) / ``W``: each stage's target is the closed-form state on k
  qubits (no seed).

Output layout::

    <base_path>/<experiment_name>/
        manifest.json
        targets/target_<k>.npy
        models/     <- --saving_path of every stage (models, warm-starts, TensorBoard)
        logs/       <- convenience folder (e.g. for SLURM stdout/stderr)

Each manifest stage records its ``run_name`` and the DETERMINISTIC path of the
warm-start it needs (``trainSB3.py`` always writes ``<run_name>_warmstart.json``
into ``--saving_path``), so the path is known before the previous stage runs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from qcsynth.states import dicke_state, random_state, schmidt_seed, w_state


# ---------------------------------------------------------------------------
# Stage-target chains
# ---------------------------------------------------------------------------
def build_schmidt_chain(target_n: np.ndarray, n_qubits: int, start_qubits: int) -> Dict[int, np.ndarray]:
    """Recursively extract the dominant Schmidt vector down to ``start_qubits`` (random targets).

    Args:
        target_n (np.ndarray): final n-qubit target statevector, shape ``(2**n_qubits,)``.
        n_qubits (int): number of qubits of ``target_n`` (largest stage).
        start_qubits (int): number of qubits of the smallest stage.

    Returns:
        Dict[int, np.ndarray]: ``{k: statevector}`` for k = n_qubits, n_qubits-1,
        ..., start_qubits; each state is the dominant Schmidt vector of the one
        above it.
    """
    chain = {n_qubits: target_n}
    psi = target_n
    k = n_qubits
    while k > start_qubits:
        phi_A, _phi_B, lambda0 = schmidt_seed(psi, k)
        k -= 1
        chain[k] = phi_A
        print(f"  Schmidt-extracted {k}-qubit sub-target from the {k + 1}-qubit one "
              f"(padding-fidelity ceiling lambda0={lambda0:.4f})")
        psi = phi_A
    return chain


def build_formula_chain(state_type: str, n_qubits: int, start_qubits: int) -> Dict[int, np.ndarray]:
    """Closed-form stage targets for W / Dicke curricula.

    Args:
        state_type (str): ``"dicke"`` (k = 2) or ``"W"``.
        n_qubits (int): number of qubits of the largest stage.
        start_qubits (int): number of qubits of the smallest stage.

    Returns:
        Dict[int, np.ndarray]: ``{k: statevector}`` for k = start_qubits, ..., n_qubits.
    """
    formula = dicke_state if state_type == "dicke" else w_state
    return {k: formula(k) for k in range(start_qubits, n_qubits + 1)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``prepare_curriculum.py``.

    Returns:
        argparse.ArgumentParser: parser with all curriculum options.
    """
    parser = argparse.ArgumentParser(
        description="Generate the full curriculum chain + manifest (random/dicke/W)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_qubits", type=int, required=True, help="Final (largest) number of qubits")
    parser.add_argument("--state_type", type=str, default="random", choices=["random", "dicke", "W"],
                        help="Which target family to generate the curriculum for.")
    parser.add_argument("--base_path", type=str, default="/hpc/archive/G_QSLAB/marco/PPO",
                        help="Root directory everything is organized under")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Subfolder under --base_path for this curriculum chain. "
                             "Default: '<state_type>_n<N>[_seed<SEED>]'")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for sampling the final target. Only used with "
                             "--state_type random (ignored with a warning otherwise). If omitted, "
                             "a seed is generated and saved in manifest.json.")
    parser.add_argument("--start_qubits", type=int, default=2,
                        help="Smallest stage, trained from scratch (no warm-start)")
    parser.add_argument("--run_label", type=str, default=None,
                        help="Optional tag prefixed to every stage's run_name, e.g. 'stage3' "
                             "becomes '<run_label>_stage3'.")
    return parser


def main() -> None:
    """Generate the stage targets and write ``manifest.json``.

    """
    args = build_parser().parse_args()

    if args.n_qubits < args.start_qubits:
        raise ValueError("--n_qubits must be >= --start_qubits")

    if args.state_type != "random" and args.seed is not None:
        print(f"WARNING: --seed is ignored for --state_type {args.state_type} "
              f"(dicke/W targets are deterministic, there is nothing to seed).")
        args.seed = None

    if args.state_type == "random" and args.seed is None:
        # Draw a seed from OS entropy ourselves so the run is still random but
        # the actual value is RECORDED and the chain can be reproduced.
        args.seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        print(f"No --seed given, generated one automatically: {args.seed} (saved in manifest.json)")

    # --- directory layout ---
    default_name = f"{args.state_type}_n{args.n_qubits}"
    if args.state_type == "random":
        default_name += f"_seed{args.seed}"
    experiment_name = args.experiment_name or default_name
    experiment_root = os.path.join(args.base_path, experiment_name)
    targets_dir = os.path.join(experiment_root, "targets")
    models_dir = os.path.join(experiment_root, "models")   # --saving_path for trainSB3.py
    logs_dir = os.path.join(experiment_root, "logs")
    for d in (targets_dir, models_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    # --- stage targets ---
    if args.state_type == "random":
        target_n = random_state(args.n_qubits, seed=args.seed)
        print(f"Sampled genuinely random {args.n_qubits}-qubit target (seed={args.seed}).")
        chain = build_schmidt_chain(target_n, args.n_qubits, args.start_qubits)
    else:
        print(f"Generating {args.state_type} targets for {args.start_qubits}..{args.n_qubits} qubits "
              f"(exact closed-form, no sampling).")
        chain = build_formula_chain(args.state_type, args.n_qubits, args.start_qubits)

    # --- stages ---
    stages = []
    prev_run_name = None
    for k in range(args.start_qubits, args.n_qubits + 1):
        target_path = os.path.join(targets_dir, f"target_{k}.npy")
        np.save(target_path, chain[k])
        run_name = f"{args.run_label}_stage{k}" if args.run_label else f"stage{k}"
        warmstart_path = (os.path.join(models_dir, f"{prev_run_name}_warmstart.json")
                          if prev_run_name is not None else None)

        stage = {
            "n_qubits": k,
            "target_path": target_path,
            "run_name": run_name,
            "warmstart_path": warmstart_path,
        }

        stages.append(stage)
        prev_run_name = run_name

    # --- manifest ---
    manifest = {
        "n_qubits_final": args.n_qubits,
        "state_type": args.state_type,
        "start_qubits": args.start_qubits,
        "base_path": args.base_path,
        "experiment_name": experiment_name,
        "experiment_root": experiment_root,
        "out_dir": models_dir,   # what --saving_path is set to for trainSB3.py
        "targets_dir": targets_dir,
        "log_dir": logs_dir,
        "seed": args.seed,
        "stages": stages,
    }
    manifest_path = os.path.join(experiment_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExperiment root: {experiment_root}")
    print(f"  targets/  -> {targets_dir}")
    print(f"  models/   -> {models_dir}")
    print(f"  logs/     -> {logs_dir}")
    print(f"\nWrote {len(stages)} stages to {manifest_path}")
    print("Now run one stage per job, in order, e.g.:")
    print(f"  python trainSB3.py --manifest {manifest_path} --n_qubits {args.start_qubits} [...hyperparams...]")
    print(f"  python trainSB3.py --manifest {manifest_path} --n_qubits {args.start_qubits + 1} [...hyperparams...]")
    if args.n_qubits > args.start_qubits + 1:
        print("  ...")
        print(f"  python trainSB3.py --manifest {manifest_path} --n_qubits {args.n_qubits} [...hyperparams...]")
    print(f"\nOr, on SLURM:  ./submit_chain.sh {manifest_path}")


if __name__ == "__main__":
    main()
