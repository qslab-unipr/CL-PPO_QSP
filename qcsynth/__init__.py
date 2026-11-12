"""
qcsynth -- reinforcement-learning quantum circuit synthesis (state preparation).

Modules:
    env          QuantumGymEnv, the gymnasium environment (RL outer loop +
                 angle-optimisation inner loop, action masking)
    states       target states (W, Dicke, random) and Schmidt decomposition
    networks     actor/critic network definition and cross-stage weight transfer
    callbacks    QuantumCallback: logging, best-circuit tracking, early stopping
    warmstart    warm-start JSON export
    manifest     curriculum manifest resolution for ``trainSB3.py --manifest``
    env_factory  CLI args -> env constructor for SubprocVecEnv
    cli          argument parser of ``trainSB3.py``
"""

from .env import QuantumGymEnv

__all__ = ["QuantumGymEnv"]
