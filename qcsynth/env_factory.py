"""
Environment factory: turns parsed CLI arguments into a picklable callable that
builds one action-masked :class:`~qcsynth.env.QuantumGymEnv` (for
``SubprocVecEnv``).
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import numpy as np
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker

from .env import QuantumGymEnv

__all__ = ["make_env_fn"]


def make_env_fn(args: argparse.Namespace) -> Callable[[], gym.Env]:
    """Return a picklable thunk that builds one action-masked environment.

    Args:
        args (argparse.Namespace): parsed ``trainSB3.py`` arguments (target,
            reward and inner-loop options).

    Returns:
        Callable[[], gym.Env]: function with no arguments returning
        ``ActionMasker(QuantumGymEnv(...))``, as required by ``SubprocVecEnv``.

    Raises:
        ValueError: if ``--state custom`` is used without ``--custom_target_path``.
    """
    env_kwargs: Dict[str, Any] = dict(
        n_qubits=args.n_qubits,
        max_gates=args.max_gates,
        threshold=args.threshold,
        state=args.state,
        variable_penalty=args.var_pen,
        variable_reward=args.var_rew,
        # inner loop (angle optimisation)
        lr=args.q_lr,
        epochs=args.q_epochs,
        penalty=args.q_penalty,
        loss_mult=args.q_loss_mult,
        clip=args.q_clip,
        es_epochs=args.es_epochs,
        exp=args.q_exp,
    )

    if args.state == "custom":
        if args.custom_target_path is None:
            raise ValueError("--state custom requires --custom_target_path <file.npy> (or use --manifest)")
        env_kwargs["custom_target"] = np.load(args.custom_target_path)
        if args.warmstart_path is not None:
            env_kwargs["warmstart_path"] = args.warmstart_path

    def _init() -> gym.Env:
        """Build one environment (called once inside every ``SubprocVecEnv`` worker).

        Returns:
            gym.Env: ``QuantumGymEnv`` wrapped in ``ActionMasker`` (exposes
            ``action_masks()`` to ``MaskablePPO``).
        """
        env = QuantumGymEnv(**env_kwargs)
        # ActionMasker exposes env.action_masks() to MaskablePPO
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    return _init
