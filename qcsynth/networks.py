"""
Policy / value network definitions and cross-stage network warm-start.

The actor and the critic are two SB3 ``MlpPolicy`` MLPs with three hidden
layers of 256 units each. This module builds the SB3 ``policy_kwargs`` and
implements the partial weight transfer used between curriculum stages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from sb3_contrib import MaskablePPO

__all__ = ["NET_ARCH", "make_policy_kwargs", "transfer_compatible_weights"]

#: Hidden-layer widths of both the actor (``pi``) and the critic (``vf``) MLPs.
NET_ARCH: List[int] = [256, 256, 256]


def make_policy_kwargs(act_fun: str) -> Dict[str, Any]:
    """Build SB3 ``policy_kwargs``: separate actor/critic MLPs with :data:`NET_ARCH` hidden layers.

    Args:
        act_fun (str): activation function, ``"relu"`` or ``"tanh"`` (anything
            other than ``"tanh"`` gives ReLU).

    Returns:
        Dict[str, Any]: ``{"net_arch": {"pi": NET_ARCH, "vf": NET_ARCH},
        "activation_fn": torch.nn.ReLU | torch.nn.Tanh}``.
    """
    act_cls = torch.nn.Tanh if act_fun == "tanh" else torch.nn.ReLU
    return dict(
        net_arch=dict(pi=NET_ARCH, vf=NET_ARCH),
        activation_fn=act_cls,
    )


def transfer_compatible_weights(model: MaskablePPO, prev_model_path: str) -> Tuple[int, int]:
    """Partial network warm-start across curriculum stages.

    Observation size (~``2**n_qubits``) and number of actions (grows with
    ``n_qubits``) differ between stage n and n+1, so the input layers and the
    policy's action head cannot be copied. The hidden layers have a fixed width
    (independent of ``n_qubits``) and the value head always outputs one scalar,
    so those tensors match. Every parameter tensor whose name AND shape match
    between the previous stage's policy and ``model.policy`` is copied;
    everything else keeps its fresh initialisation. Both stages must use the
    same ``--act_fun`` for the transfer to be meaningful.

    Args:
        model (MaskablePPO): freshly built model of the current stage (modified in place).
        prev_model_path (str): path to the previous stage's ``*_best_model.zip``.

    Returns:
        Tuple[int, int]: ``(transferred, skipped)`` numbers of parameter tensors
        (``(0, 0)`` if the previous model cannot be loaded).
    """
    try:
        prev_model = MaskablePPO.load(prev_model_path, device="cpu")
    except Exception as e:
        print(f"[network warm-start] could not load {prev_model_path}: {e} -- skipping, "
              f"this stage's network starts from scratch.")
        return 0, 0

    prev_state = prev_model.policy.state_dict()
    new_state = model.policy.state_dict()

    transferred, skipped = 0, 0
    merged = dict(new_state)
    for key, prev_tensor in prev_state.items():
        if key in new_state and new_state[key].shape == prev_tensor.shape:
            merged[key] = prev_tensor.clone()
            transferred += 1
        else:
            skipped += 1

    model.policy.load_state_dict(merged)
    print(f"[network warm-start] transferred {transferred} matching-shape parameter "
          f"tensors from {prev_model_path} ({skipped} left at fresh initialization "
          f"due to shape mismatch -- expected for input/output layers whose size "
          f"depends on n_qubits).")
    return transferred, skipped
