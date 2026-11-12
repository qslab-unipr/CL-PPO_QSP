"""
Warm-start file I/O.

A warm-start file ``<run_name>_warmstart.json`` stores the best circuit found
by a training run::

    {
      "gates":  [{"gate": "Ry", "wires": [0], "is_rot": true},
                 {"gate": "CNOT", "wires": [0, 1], "is_rot": false}, ...],
      "params": [0.785398, ...],        # one angle per rotation gate, in order
      "dist":   3.1e-11,                # distance reached by this circuit
      "model_path": ".../stage3_best_model.zip"   # optional
    }

The next curriculum stage replays ``gates``/``params`` at the start of every
episode (circuit warm-start) and, if ``model_path`` is present, copies the
shape-compatible network weights from it (network warm-start).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["BestCircuit", "export_warmstart"]

#: ``(gate_list, params, dist)`` -- the best circuit found so far.
BestCircuit = Tuple[List[Dict[str, Any]], Sequence[float], float]


def export_warmstart(saving_path: str, run_name: str, best_circuit: BestCircuit,
                     model_path: Optional[str] = None) -> str:
    """Write ``<saving_path>/<run_name>_warmstart.json``.

    Called on every new best circuit, at every periodic checkpoint, on SIGTERM
    and at the end of training, so a job killed by the scheduler still leaves a
    usable warm-start on disk.

    Args:
        saving_path (str): output directory.
        run_name (str): run name, used as the file prefix.
        best_circuit (BestCircuit): ``(gate_list, params, dist)``: the gate dicts
            (List[Dict[str, Any]]), their rotation angles (Sequence[float]) and
            the infidelity reached (float).
        model_path (Optional[str]): path of the matching ``*_best_model.zip``,
            stored so the next stage can warm-start its network; omitted if None.

    Returns:
        str: path of the written JSON file.
    """
    gate_list, params, dist = best_circuit
    warmstart_out = os.path.join(saving_path, f"{run_name}_warmstart.json")
    payload: Dict[str, Any] = {
        "gates": gate_list,
        "params": [float(p) for p in np.asarray(params)],
        "dist": float(dist),
    }
    if model_path is not None:
        payload["model_path"] = model_path
    with open(warmstart_out, "w") as f:
        json.dump(payload, f, indent=2)
    return warmstart_out
