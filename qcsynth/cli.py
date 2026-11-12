"""
Command-line interface of ``trainSB3.py``.

Every default is the configuration used in the paper, so a curriculum stage
can be run with just ``--manifest``, ``--n_qubits`` and ``--max_gates``.
Boolean options that are ON by default have a matching ``--no_<option>`` flag.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

__all__ = ["build_parser", "get_args"]


def _add_on_by_default(group: argparse._ArgumentGroup, name: str, help_on: str) -> None:
    """Add a boolean option that is ON by default, with a ``--no_<name>`` switch.

    Args:
        group (argparse._ArgumentGroup): argument group to add the options to.
        name (str): option name without dashes; also the destination attribute.
        help_on (str): help text of ``--<name>``.

    Returns:
        None
    """
    group.add_argument(f"--{name}", dest=name, action="store_true", default=True, help=help_on)
    group.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable --{name}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``trainSB3.py`` (defaults = paper configuration).

    Returns:
        argparse.ArgumentParser: parser with all training options.
    """
    parser = argparse.ArgumentParser(
        description="SB3 MaskablePPO – Quantum Circuit Synthesis (defaults = paper configuration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Run / I/O
    # ------------------------------------------------------------------
    g = parser.add_argument_group("run / I/O")
    g.add_argument("--run_name", type=str, default="run_sb3",
                   help="Prefix of every file this run writes (overridden by --manifest)")
    g.add_argument("--saving_path", type=str, default="./",
                   help="Output directory for models, warm-starts and TensorBoard logs "
                        "(overridden by --manifest)")
    g.add_argument("--load_model", action="store_true",
                   help="Resume from <saving_path>/<run_name>_final.zip (and its VecNormalize stats)")
    g.add_argument("--test", action="store_true",
                   help="Evaluation mode: run the policy deterministically for 1000 vectorised "
                        "steps and print dist/gate count of finished episodes (no training)")

    # ------------------------------------------------------------------
    # Environment / target
    # ------------------------------------------------------------------
    g = parser.add_argument_group("environment / target")
    g.add_argument("--n_qubits", type=int, default=2,
                   help="Number of qubits (with --manifest: which stage to run)")
    g.add_argument("--max_gates", type=int, default=30,
                   help="Episode is truncated after this many gates (including the replayed "
                        "warm-start circuit)")
    g.add_argument("--threshold", type=float, default=1e-2,
                   help="Success threshold on the infidelity: success when 1 - F < threshold")
    g.add_argument("--state", type=str, default="custom", choices=["W", "dicke", "custom"],
                   help="Target: 'W', 'dicke' (k=2) or 'custom' (--custom_target_path). "
                        "Forced to 'custom' by --manifest.")
    g.add_argument("--custom_target_path", type=str, default=None,
                   help="Path to a .npy statevector, used when --state custom "
                        "(set automatically by --manifest)")
    g.add_argument("--warmstart_path", type=str, default=None,
                   help="Path to a <run_name>_warmstart.json of an already-solved (n-1)-qubit "
                        "circuit, replayed at the start of every episode (curriculum learning). "
                        "Only used with --state custom. Set automatically by --manifest.")
    g.add_argument("--manifest", type=str, default=None,
                   help="Path to a manifest.json from prepare_curriculum.py. The stage matching "
                        "--n_qubits is looked up and used to set --state=custom, "
                        "--custom_target_path, --warmstart_path, --run_name and --saving_path. "
                        "Refuses to start if the previous stage's warmstart file is missing.")
    _add_on_by_default(g, "force",
                       "With --manifest: retrain even if <run_name>_final.zip already exists "
                       "(--no_force skips stages that are already done)")

    # ------------------------------------------------------------------
    # Reward (log_pen)
    # ------------------------------------------------------------------
    g = parser.add_argument_group(
        "reward (log_pen)",
        "r = q_loss_mult * [log d_prev - log d] + q_penalty, plus the --var_rew bonuses")
    g.add_argument("--q_penalty", type=float, default=-1, help="Per-step penalty")
    g.add_argument("--q_loss_mult", type=int, default=10,
                   help="Multiplier of the log-distance improvement")
    _add_on_by_default(g, "var_rew",
                       "Milestone bonuses (+20 the first time d < 1e-1, 1e-2, ... down to --threshold) and "
                       "exponential record bonus on success")
    g.add_argument("--q_exp", type=int, default=75,
                   help="Scale of the exponential record bonus (--var_rew)")
    g.add_argument("--var_pen", action="store_true",
                   help="Scale the per-step penalty by the current gate count")
    g.add_argument("--q_clip", action="store_true",
                   help="Clip the step reward to [-10, 10] (before the --var_rew bonuses)")

    # ------------------------------------------------------------------
    # Inner loop: circuit-angle optimisation
    # ------------------------------------------------------------------
    g = parser.add_argument_group("inner loop (angle optimisation with PyTorch Adam)")
    g.add_argument("--q_lr", type=float, default=0.05, help="Inner Adam learning rate")
    g.add_argument("--q_epochs", type=int, default=200, help="Max inner iterations per RL step")
    g.add_argument("--es_epochs", type=int, default=20,
                   help="Patience (inner iterations without improvement) of the inner early stopping")
    # ------------------------------------------------------------------
    # PPO
    # ------------------------------------------------------------------
    g = parser.add_argument_group("PPO / SB3")
    g.add_argument("--num_envs", type=int, default=16, help="Parallel environments (SubprocVecEnv)")
    g.add_argument("--steps", type=int, default=128, help="n_steps per env per rollout (SB3 n_steps)")
    g.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate")
    g.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    g.add_argument("--eps_clip", type=float, default=0.2, help="SB3 clip_range")
    g.add_argument("--epochs", type=int, default=10, help="SB3 n_epochs")
    g.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    g.add_argument("--act_fun", type=str, default="relu", choices=["tanh", "relu"],
                   help="Activation function")
    _add_on_by_default(g, "norm_reward",
                       "Wrap env with VecNormalize(norm_reward=True) to stabilize PPO with uneven "
                       "reward scales")
    g.add_argument("--clip_reward", type=float, default=10.0,
                   help="Clip normalized rewards to [-clip_reward, +clip_reward] (VecNormalize)")

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    # Measured in TIMESTEPS, not episodes: a failed episode costs up to
    # --max_gates steps while a successful one can be very short, so episode
    # counts would mean very different amounts of training per stage.
    # The success rate is always computed over the last 100 episodes.
    g = parser.add_argument_group("early stopping (all patience values in timesteps)")
    _add_on_by_default(g, "early_stop", "Enable early stopping (see the three triggers below)")
    g.add_argument("--early_stop_min_steps", type=int, default=50_000,
                   help="Minimum timesteps before early stopping is even considered")
    g.add_argument("--early_stop_patience_steps", type=int, default=10_000,
                   help="Soft trigger (paper rule): timesteps without a shorter circuit, AND success "
                        "rate over the last 100 episodes >= --early_stop_min_success_rate")
    g.add_argument("--early_stop_min_success_rate", type=float, default=0.98,
                   help="Required success rate over the last 100 episodes for the soft trigger")
    g.add_argument("--early_stop_hard_patience_steps", type=int, default=450_000,
                   help="Hard trigger: stop after this many timesteps without a shorter circuit, "
                        "regardless of success rate. Pass a huge value to effectively disable it.")
    g.add_argument("--early_stop_plateau_window_steps", type=int, default=220_000,
                   help="Plateau trigger: stop if the 100-episode success rate has improved by less "
                        "than --early_stop_plateau_min_delta over this many timesteps. Pass a huge "
                        "value to effectively disable it.")
    g.add_argument("--early_stop_plateau_min_delta", type=float, default=0.02,
                   help="Minimum success-rate improvement over the plateau window "
                        "(0.02 = 2 percentage points)")

    # ------------------------------------------------------------------
    # Network warm-start
    # ------------------------------------------------------------------
    g = parser.add_argument_group("network warm-start across curriculum stages")
    g.add_argument("--best_model_min_interval", type=int, default=2000,
                   help="Minimum timesteps between saves of <run_name>_best_model.zip (written "
                        "whenever a new best circuit is found)")
    _add_on_by_default(g, "network_warmstart",
                       "Copy shape-matching policy/value weights from the previous stage's "
                       "best_model.zip (found via --manifest or --prev_model_path) into the new "
                       "network. Input layer and action head change shape with n_qubits and stay "
                       "freshly initialised.")
    g.add_argument("--prev_model_path", type=str, default=None,
                   help="Previous stage's <run_name>_best_model.zip, for manual use without "
                        "--manifest (with --manifest it is read from the previous warmstart JSON "
                        "and this value is overridden)")

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------
    g = parser.add_argument_group("training budget")
    g.add_argument("--total_steps", type=int, default=1_000_000_000,
                   help="Total env timesteps (in practice early stopping or the job wall-clock, "
                        "3 days in the paper, ends the run)")
    g.add_argument("--save_every", type=int, default=50_000,
                   help="Save a checkpoint every N env timesteps")

    return parser


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the ``trainSB3.py`` command line.

    Args:
        argv (Optional[List[str]]): arguments to parse; ``None`` uses ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: parsed options.
    """
    return build_parser().parse_args(argv)
