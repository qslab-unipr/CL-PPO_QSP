"""
Training callback: TensorBoard logging, best-circuit tracking, checkpointing
and early stopping.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .warmstart import BestCircuit, export_warmstart

__all__ = ["QuantumCallback"]


class QuantumCallback(BaseCallback):
    """SB3 callback for quantum-circuit-synthesis runs.

    At the end of every episode it:

    * logs circuit quality, success rates and rewards under ``quantum/*``;
    * keeps the **best circuit** (fewest gates; ties broken by lower
      infidelity). Failed (truncated) episodes are also candidates, so before
      the first success the "best" circuit is simply the best one seen;
    * on every new best circuit, writes ``<run_name>_warmstart.json`` and
      (throttled by ``--best_model_min_interval``) ``<run_name>_best_model.zip``;
    * with ``--early_stop`` (on by default), stops training when one of the
      triggers described in :meth:`_check_early_stop` fires.

    Every ``--save_every`` timesteps it saves ``<run_name>_step<N>.zip`` and
    refreshes the warm-start file. Constructor arguments are documented in
    :meth:`__init__`.
    """

    def __init__(self, args: argparse.Namespace, target_psi: np.ndarray, verbose: int = 0) -> None:
        """Initialise the counters, the best-circuit tracker and the early-stopping state.

        Args:
            args (argparse.Namespace): parsed ``trainSB3.py`` arguments (see
                :mod:`qcsynth.cli`); the callback reads the threshold, saving
                options and early-stopping options from it.
            target_psi (np.ndarray): target statevector of the run (stored for reference only).
            verbose (int): SB3 verbosity level.

        Returns:
            None
        """
        super().__init__(verbose)
        self.args = args
        self.target_psi = target_psi
        self._last_save = 0
        self.n_episodes = 0
        self.n_successes = 0
        self.best_dist = 1.0
        self.n_consecutive_successes = 0
        self.recent_outcomes: Deque[int] = deque(maxlen=100)

        self.best_gate_count: Optional[int] = None
        # self.best_circuit: BestCircuit and self.best_model_path: str are
        # created lazily on the first improvement (code checks hasattr()).

        # Absolute timestep of the last strictly-shorter circuit (early-stopping
        # patience is measured in timesteps, not episodes).
        self._last_improvement_step = 0

        # (timestep, success_rate_100ep) samples, used by the plateau trigger.
        self._sr_history: Deque[Tuple[int, float]] = deque()

        # Last time best_model.zip was written (network warm-start source).
        self._last_best_model_save_step = -10**9  # force a save on the first improvement

    def _check_early_stop(self) -> bool:
        """Decide whether training should stop. Called at every episode end.

        Nothing is checked before ``--early_stop_min_steps`` timesteps. Then training
        stops when any trigger fires (success rate = over the last 100 episodes):

        * soft (paper rule): no strictly shorter circuit for
          ``--early_stop_patience_steps`` AND success rate >= ``--early_stop_min_success_rate``;
        * hard: no strictly shorter circuit for ``--early_stop_hard_patience_steps``,
          regardless of success rate;
        * plateau: the success rate improved by less than
          ``--early_stop_plateau_min_delta`` over the last
          ``--early_stop_plateau_window_steps`` timesteps.

        Appends to ``self._sr_history`` and prints the reason when stopping.

        Returns:
            bool: True if training should stop.
        """
        if not (self.args.early_stop and self.num_timesteps >= self.args.early_stop_min_steps):
            return False

        steps_since_improvement = self.num_timesteps - self._last_improvement_step
        success_rate_100ep = np.mean(self.recent_outcomes)

        soft_stop = (steps_since_improvement >= self.args.early_stop_patience_steps
                     and success_rate_100ep >= self.args.early_stop_min_success_rate)
        hard_stop = steps_since_improvement >= self.args.early_stop_hard_patience_steps

        # plateau: compare with the sample closest to `window` timesteps ago
        window = self.args.early_stop_plateau_window_steps
        self._sr_history.append((self.num_timesteps, success_rate_100ep))
        while self._sr_history and self.num_timesteps - self._sr_history[0][0] > 2 * window:
            self._sr_history.popleft()  # keep at most 2 windows of history
        plateau_stop = False
        plateau_delta = None
        if self._sr_history and self.num_timesteps - self._sr_history[0][0] >= window:
            target_t = self.num_timesteps - window
            sr_before = min(self._sr_history, key=lambda x: abs(x[0] - target_t))[1]
            plateau_delta = success_rate_100ep - sr_before
            plateau_stop = plateau_delta < self.args.early_stop_plateau_min_delta

        if not (soft_stop or hard_stop or plateau_stop):
            return False

        if soft_stop:
            reason = (f"success_rate_100ep={success_rate_100ep:.3f} >= "
                      f"{self.args.early_stop_min_success_rate}")
        elif hard_stop:
            reason = (f"exceeded --early_stop_hard_patience_steps="
                      f"{self.args.early_stop_hard_patience_steps} regardless of "
                      f"success_rate_100ep={success_rate_100ep:.3f}")
        else:
            reason = (f"success_rate plateau: only {plateau_delta:+.3f} change over the last "
                      f"{window} timesteps (< {self.args.early_stop_plateau_min_delta} required)")
        print(f"\n[EARLY STOP] No shorter/better circuit found in the last "
              f"{steps_since_improvement} timesteps (best: "
              f"{len(self.best_circuit[0])} gates, dist={self.best_circuit[2]:.2e}); "
              f"{reason}. Stopping training.\n")
        return True

    def _on_step(self) -> bool:
        """Called by SB3 after every vectorised env step.

        Processes the episodes that just ended (logging, best-circuit tracking,
        warm-start export, early stopping) and saves the periodic checkpoint.

        Returns:
            bool: False to stop training (early stopping), True to continue.
        """
        n = self.num_timesteps
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" not in info:  # VecMonitor adds "episode" only at episode end
                continue

            self.n_episodes += 1
            ep = info["episode"]
            dist = info.get("dist", 1.0)
            gate_count = info.get("gate_count", 0)
            success = dist < self.args.threshold

            # update trackers
            self.best_dist = min(self.best_dist, dist)
            self.recent_outcomes.append(int(success))
            if success:
                self.n_successes += 1
                self.n_consecutive_successes += 1
            else:
                self.n_consecutive_successes = 0

            # --- circuit quality ---
            self.logger.record("quantum/infidelity", dist)
            self.logger.record("quantum/log10_dist", np.log10(dist + 1e-16))
            self.logger.record("quantum/best_dist_ever", self.best_dist)
            self.logger.record("quantum/gate_count", gate_count)
            if success:
                self.logger.record("quantum/gates_on_success", gate_count)

            # --- success tracking ---
            self.logger.record("quantum/n_episodes", self.n_episodes)
            self.logger.record("quantum/n_successes", self.n_successes)
            self.logger.record("quantum/success_rate_total", self.n_successes / self.n_episodes)
            self.logger.record("quantum/success_rate_100ep", np.mean(self.recent_outcomes))
            self.logger.record("quantum/consecutive_successes", self.n_consecutive_successes)

            # --- reward ---
            self.logger.record("quantum/ep_reward", ep["r"])
            self.logger.record("quantum/reward_per_gate", ep["r"] / max(gate_count, 1))

            # --- best-circuit tracking ---
            gate_list = info.get("gate_list", [])
            params = info.get("params", [])

            is_new_best = (not hasattr(self, 'best_circuit')
                           or gate_count < len(self.best_circuit[0])
                           or (gate_count == len(self.best_circuit[0]) and dist < self.best_circuit[2]))
            # Early-stop patience resets ONLY on a strictly shorter circuit --
            # a same-length circuit with better fidelity still updates
            # best_circuit (it's a better warm-start seed) but does NOT count
            # as progress for early stopping.
            is_shorter = (not hasattr(self, 'best_circuit')
                          or gate_count < len(self.best_circuit[0]))

            if is_new_best:
                self.best_circuit: BestCircuit = (gate_list, params, dist)
                print(f"New shortest circuit: {gate_count} gates, dist={dist}")
                self.best_gate_count = gate_count

                p_idx = 0
                for g in gate_list:
                    if g['is_rot']:
                        print(f"  {g['gate']}(wire={g['wires'][0]}, angle={float(params[p_idx]):.6f} rad  [{float(params[p_idx])*180/np.pi:.2f}°])")
                        p_idx += 1
                    else:
                        print(f"  {g['gate']}(ctrl={g['wires'][0]}, tgt={g['wires'][1]})")
                print("=" * 50 + "\n")

                # Network warm-start: snapshot the policy that JUST found this
                # improvement. Throttled so rapid early improvements don't
                # spam disk I/O.
                if self.num_timesteps - self._last_best_model_save_step >= self.args.best_model_min_interval:
                    best_model_path = os.path.join(self.args.saving_path, f"{self.args.run_name}_best_model.zip")
                    self.model.save(best_model_path)
                    self._last_best_model_save_step = self.num_timesteps
                    self.best_model_path = best_model_path

                # Circuit warm-start: cheap JSON, written on every improvement.
                ws_path = export_warmstart(self.args.saving_path, self.args.run_name, self.best_circuit,
                                           model_path=getattr(self, "best_model_path", None))
                print(f"  warmstart saved to {ws_path}\n")

            if is_shorter:
                self._last_improvement_step = self.num_timesteps

            if self._check_early_stop():
                return False

        # --- periodic checkpoint ---
        if n - self._last_save >= self.args.save_every:
            self._last_save = n
            path = os.path.join(self.args.saving_path,
                                f"{self.args.run_name}_step{n}.zip")
            self.model.save(path)
            msg = f"[step {n} | ep {self.n_episodes} | best dist {self.best_dist:.2e}] → {path}"
            if hasattr(self, "best_circuit"):
                model_path = getattr(self, "best_model_path", None)
                ws_path = export_warmstart(self.args.saving_path, self.args.run_name, self.best_circuit, model_path=model_path)
                msg += f"  (warmstart updated: {len(self.best_circuit[0])} gates, dist={self.best_circuit[2]:.2e} → {ws_path})"
            print(msg)

        return True
