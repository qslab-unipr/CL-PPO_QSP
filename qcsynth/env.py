"""
Gymnasium environment for RL-driven quantum circuit synthesis
(quantum state preparation).

Two nested loops:

* **outer loop (RL)** – at every step the agent picks a *discrete* action: a
  gate type plus the wire(s) it acts on (``Rx``/``Ry``/``Rz`` on one qubit, or
  ``CNOT`` on an ordered pair). The agent never chooses angles.
* **inner loop (classical optimisation)** – after each new gate, all rotation
  angles of the circuit are jointly optimised with PyTorch Adam to minimise
  the distance (infidelity 1 - F) between the circuit's output state and the
  target state. The resulting distance drives the reward and the new
  statevector is part of the next observation.

An episode *terminates* when the distance drops below ``threshold`` and is
*truncated* when ``max_gates`` gates have been placed.

The env exposes :meth:`QuantumGymEnv.action_masks`, so it plugs directly into
``sb3_contrib.MaskablePPO``.

Configuration fixed to the one used in the paper: ``log_pen`` reward and
``stepsnorm_res`` observation.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pennylane as qml
from gymnasium import spaces
from pennylane import numpy as pnp

from . import states
import torch

__all__ = ["QuantumGymEnv", "Gate"]


_TORCH_THREADS_LIMITED = False


def _limit_torch_threads_once() -> None:
    """Set ``torch.set_num_threads(1)`` the first time it is called in a process.

    Returns:
        None
    """
    global _TORCH_THREADS_LIMITED
    if not _TORCH_THREADS_LIMITED:
        torch.set_num_threads(1)
        _TORCH_THREADS_LIMITED = True

#: A gate is a dict ``{"gate": str, "wires": list[int], "is_rot": bool}``.
Gate = Dict[str, Any]

def _queue_gates(gates: Sequence[Gate], angles: Any) -> None:
    """Queue ``gates`` as PennyLane operations inside an active QNode.

    Args:
        gates (Sequence[Gate]): circuit to apply, in order. Each gate is a dict
            ``{"gate": "Rx" | "Ry" | "Rz" | "CNOT" | "CZ", "wires": list[int], "is_rot": bool}``.
        angles (Any): rotation angles (numpy array, PennyLane array or torch
            tensor), one per rotation gate, consumed in order.

    Returns:
        None
    """
    p_idx = 0
    for g in gates:
        if g['gate'] == 'Rx':
            qml.RX(angles[p_idx], wires=g['wires'])
            p_idx += 1
        elif g['gate'] == 'Ry':
            qml.RY(angles[p_idx], wires=g['wires'])
            p_idx += 1
        elif g['gate'] == 'Rz':
            qml.RZ(angles[p_idx], wires=g['wires'])
            p_idx += 1
        elif g['gate'] == 'CNOT':
            qml.CNOT(wires=g['wires'])
        elif g['gate'] == 'CZ':
            qml.CZ(wires=g['wires'])


class QuantumGymEnv(gym.Env):
    """RL environment for quantum state preparation with a PyTorch-Adam
    angle-optimisation inner loop.

    Observation: ``stepsnorm_res`` (see :meth:`_get_obs`); action: index into
    ``actions_map`` (gate type + wires); reward: ``log_pen`` (see
    :meth:`_compute_reward`). Exposes :meth:`action_masks` for sb3-contrib
    ``MaskablePPO``. Constructor arguments are documented in :meth:`__init__`.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_qubits: int = 2, max_gates: int = 30, threshold: float = 1e-2,
                 state: str = 'custom',
                 lr: float = 0.05, epochs: int = 200, penalty: float = -1,
                 loss_mult: float = 10,
                 variable_penalty: bool = False, variable_reward: bool = True,
                 clip: bool = False, es_epochs: int = 20,
                 exp: float = 75,
                 custom_target: Optional[np.ndarray] = None,
                 warmstart_path: Optional[str] = None) -> None:
        """Build the action/observation spaces, the target state and the initial circuit.

        All defaults are the configuration used in the paper.

        Args:
            n_qubits (int): number of qubits.
            max_gates (int): the episode is truncated once the circuit has this many
                gates (the replayed warm-start circuit included).
            threshold (float): success threshold on the infidelity ``1 - F``, with
                ``F = |<target|psi>|^2``.
            state (str): target state: ``"W"``, ``"dicke"`` (k = 2) or ``"custom"``
                (requires ``custom_target``).
            lr (float): learning rate of the inner Adam optimiser.
            epochs (int): maximum number of inner Adam iterations per RL step.
            penalty (float): per-step penalty added to the reward (negative).
            loss_mult (float): multiplier of the log-infidelity improvement in the reward.
            variable_penalty (bool): if True, the step penalty is multiplied by the
                current gate count.
            variable_reward (bool): if True, add the milestone bonuses (+20 at
                1e-1, 1e-2, ... down to ``threshold``) and the exponential record
                bonus on success (scaled by ``exp``).
            clip (bool): if True, clip the per-step reward to [-10, 10] before the
                bonuses are added.
            es_epochs (int): patience of the inner early stopping (iterations without
                a relative improvement), see :meth:`check_early_stopping`.
            exp (float): scale of the exponential record bonus.
            custom_target (Optional[np.ndarray]): target statevector of shape
                ``(2**n_qubits,)``, used (and renormalised) when ``state == "custom"``.
            warmstart_path (Optional[str]): path to a warm-start JSON
                ``{"gates": [...], "params": [...]}`` of an already-solved circuit
                (typically on n-1 qubits), replayed at the start of every episode.

        Returns:
            None
        """
        super().__init__()

        # --- hyper-parameters ---
        self.n_qubits = n_qubits
        self.max_gates = max_gates
        self.threshold = threshold
        self.lr = lr
        self.epochs = epochs
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.variable_penalty = variable_penalty
        self.variable_reward = variable_reward
        self.state = state
        self.clip = clip
        self.es_epochs = es_epochs
        self.exp = exp

        # --- curriculum-learning support ---
        # custom_target: statevector used when state == "custom" (e.g. the
        #                dominant Schmidt vector of the next stage's target,
        #                produced by prepare_curriculum.py)
        # warmstart_path: JSON with {"gates": [...], "params": [...]} of an
        #                 already-solved (n-1)-qubit circuit (written by
        #                 trainSB3.py as <run_name>_warmstart.json)
        self._custom_target = custom_target
        self.warmstart_path = warmstart_path

        # shortest successful circuit seen by THIS env instance (used by the
        # exponential record bonus of the log_pen + variable_reward scheme)
        self.best_gate_count_ever: Optional[int] = None

        # qubit_active[q] is False while qubit q is still in |0>: a CNOT
        # controlled on it would be a no-op, so it is masked out.
        self.qubit_active: List[bool] = [False] * n_qubits
        self.last_action_idx = -1

        # distance milestones already rewarded in this episode
        self.reached_milestones: set = set()

        self.dev = qml.device("default.qubit", wires=n_qubits)

        # --- action space: gate type + wires only (no angles!) ---
        # indices [0, 3n): Rx/Ry/Rz on each qubit; then CNOT on every ordered pair.
        self.actions_map: List[Gate] = []
        for g in ['Rx', 'Ry', 'Rz']:
            for q in range(n_qubits):
                self.actions_map.append({'gate': g, 'wires': [q], 'is_rot': True})
        for g in ['CNOT']:
            for i in range(n_qubits):
                for j in range(n_qubits):
                    if i != j:
                        self.actions_map.append({'gate': g, 'wires': [i, j], 'is_rot': False})
        self.action_dim = len(self.actions_map)

        # --- observation space (stepsnorm_res) ---
        # [current state (re, im), gate_count / max_gates, target - current (re, im)]
        self.state_dim = 4 * (2 ** n_qubits) + 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)

        # --- target state ---
        print(f"--- Generating Fixed Target State for {n_qubits} Qubits ---")
        self.choose_state()
        with np.printoptions(precision=18):
            print(self.target_psi)

        self._reset_internal()

    # ------------------------------------------------------------------
    # Target state selection
    # ------------------------------------------------------------------
    def choose_state(self) -> None:
        """Set ``self.target_psi`` and ``self.target_rho`` from ``self.state``.

        Returns:
            None

        Raises:
            ValueError: if ``self.state`` is unknown, or ``"custom"`` without a
                target / with a target of the wrong dimension.
        """
        if self.state == "W":
            self.target_psi = states.w_state(self.n_qubits)
        elif self.state == "dicke":
            self.target_psi = states.dicke_state(self.n_qubits, k=2)
        elif self.state == "custom":
            if self._custom_target is None:
                raise ValueError("state == 'custom' requires custom_target=<statevector> to be passed to QuantumGymEnv")
            psi = np.asarray(self._custom_target, dtype=np.complex128).flatten()
            expected_dim = 2 ** self.n_qubits
            if psi.shape[0] != expected_dim:
                raise ValueError(f"custom_target has dim {psi.shape[0]}, expected {expected_dim} for n_qubits={self.n_qubits}")
            # renormalize defensively (Schmidt vectors from SVD are already unit-norm,
            # but this guards against float drift / externally-supplied targets)
            self.target_psi = (psi / np.linalg.norm(psi)).astype(np.complex128)
        else:
            raise ValueError(f"Unknown state {self.state!r}: expected 'W', 'dicke' or 'custom'")

        self.target_rho = np.outer(self.target_psi, np.conj(self.target_psi))

    # Kept as a static method for backward compatibility
    # (``QuantumGymEnv.schmidt_seed(psi, n)``).
    schmidt_seed = staticmethod(states.schmidt_seed)

    # ------------------------------------------------------------------
    # Internal reset (used by __init__ and reset())
    # ------------------------------------------------------------------
    def _reset_internal(self) -> np.ndarray:
        """Reset the circuit to |0...0>, replaying the warm-start circuit if one is set.

        Also resets ``prev_dist``, the active-qubit flags and the reached milestones.

        Returns:
            np.ndarray: initial observation, shape ``(4 * 2**n_qubits + 1,)``, dtype ``float32``.
        """
        self.current_gates: List[Gate] = []
        self.params = pnp.array([], requires_grad=True)
        self.prev_dist = 1.0

        # Initial state is |0...0>
        init_psi = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
        init_psi[0] = 1.0

        self.qubit_active = [False] * self.n_qubits
        self.last_action_idx = -1
        self.reached_milestones = set()

        if self.warmstart_path is not None:
            # Uniform warm-start mechanism for ANY target: the JSON holds the
            # gate list + params of an already-solved (n-1)-qubit circuit
            # (found by RL in the previous curriculum stage). We simply replay
            # it; the new qubit is left in |0>, and the agent has to learn how
            # to rotate/entangle it.
            with open(self.warmstart_path) as f:
                ws = json.load(f)

            self.current_gates = [dict(g) for g in ws["gates"]]
            self.params = pnp.array(ws["params"], requires_grad=True)

            init_psi = self._run_circuit(self.params)
            fid = np.abs(np.vdot(init_psi, self.target_psi)) ** 2
            self.prev_dist = 1 - fid

            for i in range(self.n_qubits - 1):
                self.qubit_active[i] = True

        return self._get_obs(init_psi)

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode (gymnasium API).

        Args:
            seed (Optional[int]): seed forwarded to ``gym.Env.reset`` (seeds
                ``self.np_random``; the env itself draws from numpy's global RNG).
            options (Optional[Dict[str, Any]]): unused, required by the gymnasium API.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: ``(observation, info)``; the
            observation has shape ``(4 * 2**n_qubits + 1,)`` and dtype ``float32``,
            ``info`` is an empty dict.
        """
        super().reset(seed=seed)
        self.applied_action_indices: List[int] = []
        obs = self._reset_internal()
        return obs.astype(np.float32), {}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Append the gate ``actions_map[action_idx]``, re-optimise all angles and score the result.

        Args:
            action_idx (int): index of the chosen action in ``self.actions_map``.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            ``(obs, reward, terminated, truncated, info)`` where

            * ``obs`` (np.ndarray, float32): observation of the new state;
            * ``reward`` (float): ``log_pen`` reward of this step;
            * ``terminated`` (bool): True if the infidelity is below ``threshold``;
            * ``truncated`` (bool): True if ``max_gates`` was reached without success;
            * ``info`` (Dict[str, Any]): ``dist`` (float, infidelity) and
              ``gate_count`` (int); at episode end also ``gate_list``
              (List[Gate], the whole circuit) and ``params`` (array of angles).
        """
        action_idx = int(action_idx)
        self.last_action_idx = action_idx
        action = self.actions_map[action_idx]
        self.current_gates.append(action)

        # Add a new (near-zero) angle if the gate is a rotation
        if action['is_rot']:
            self.params = pnp.append(self.params, pnp.array([np.random.normal(0, 0.01)], requires_grad=True))
            self.params = pnp.array(self.params, requires_grad=True)
            self.qubit_active[action['wires'][0]] = True
        else:
            self.qubit_active[action['wires'][1]] = True  # CNOT target

        self.applied_action_indices.append(action_idx)

        # Optimize angles for the new structure
        dist, opt_psi = self._optimize_circuit()
        dist = float(np.clip(dist, 1e-16, 1.0))

        reward = self._compute_reward(dist)

        terminated = bool(dist < self.threshold)
        truncated = bool(len(self.current_gates) >= self.max_gates) and not terminated

        if terminated:
            if self.best_gate_count_ever is None or self.best_gate_count_ever > len(self.current_gates):
                self.best_gate_count_ever = len(self.current_gates)

        obs = self._get_obs(opt_psi).astype(np.float32)

        info: Dict[str, Any] = {
            "dist": dist,
            "gate_count": len(self.current_gates),
        }
        if terminated or truncated:
            info["gate_list"] = list(self.current_gates)
            info["params"] = self.params

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, dist: float) -> float:
        """``log_pen`` reward for reaching infidelity ``dist`` after the current step.

        ``loss_mult * [log(prev_dist) - log(dist)] + penalty``: the improvement in
        log-infidelity minus a per-step penalty (scaled by the gate count if
        ``variable_penalty``), optionally clipped to [-10, 10] (``clip``). With
        ``variable_reward``:

        * +20 the first time (within the episode) ``dist`` drops below each order
          of magnitude 1e-1, 1e-2, ... down to ``threshold`` (included);
        * on success, ``exp * exp(-0.5 * max(gap, -2))`` where ``gap`` is the gate
          count minus the best gate count reached so far by this env (10 for the
          first success).

        Updates ``self.prev_dist`` and ``self.reached_milestones``.

        Args:
            dist (float): infidelity ``1 - F`` reached after this step.

        Returns:
            float: reward of the step.
        """
        epsilon = 1e-15

        delta = -np.log(dist + epsilon) - (-np.log(self.prev_dist + epsilon))
        self.prev_dist = dist

        if not self.variable_penalty:
            step_penalty = self.penalty
        else:
            step_penalty = self.penalty * len(self.current_gates)
        reward = delta * self.loss_mult + step_penalty

        if self.clip:
            reward = np.clip(reward, -10, 10)

        if self.variable_reward:
            # milestone bonuses at 1e-1, 1e-2, ... down to the threshold (included)
            for i in range(1, round(-math.log10(self.threshold)) + 1):
                m = 10 ** (-i)
                if dist < m and m not in self.reached_milestones:
                    reward += 20
                    self.reached_milestones.add(m)

            if dist < self.threshold:
                if self.best_gate_count_ever is not None:
                    gap_from_best = len(self.current_gates) - self.best_gate_count_ever
                    # gap_from_best == 0 -> matched best; negative -> new record
                    exp_bonus = self.exp * np.exp(-0.5 * max(gap_from_best, -2))
                else:
                    exp_bonus = 10
                reward += exp_bonus

        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self, current_psi: np.ndarray) -> np.ndarray:
        """Build the ``stepsnorm_res`` observation for ``current_psi``.

        Layout: ``[Re(psi), Im(psi), gate_count / max_gates, Re(target - psi), Im(target - psi)]``.

        Args:
            current_psi (np.ndarray): current circuit output state, shape ``(2**n_qubits,)``.

        Returns:
            np.ndarray: observation, shape ``(4 * 2**n_qubits + 1,)``, dtype ``float32``.
        """
        target_flat = np.concatenate([
            np.array(self.target_psi, dtype=np.complex128).real,
            np.array(self.target_psi, dtype=np.complex128).imag
        ])
        curr_flat = np.concatenate([current_psi.real, current_psi.imag])
        steps_used = np.array([len(self.current_gates) / self.max_gates])
        residual = target_flat - curr_flat
        return np.concatenate([curr_flat, steps_used, residual]).astype(np.float32)

    # ------------------------------------------------------------------
    # Circuit simulation helpers
    # ------------------------------------------------------------------
    def _run_circuit(self, angles: Any) -> np.ndarray:
        """Simulate the current circuit (``self.current_gates``) with PennyLane.

        Args:
            angles (Any): rotation angles, one per rotation gate (array-like).

        Returns:
            np.ndarray: output statevector, shape ``(2**n_qubits,)``, dtype ``complex128``.
        """
        @qml.qnode(self.dev)
        def circuit():
            """QNode applying ``self.current_gates`` with the enclosing ``angles``.

            Returns:
                np.ndarray: output statevector, shape ``(2**n_qubits,)``.
            """
            _queue_gates(self.current_gates, angles)
            return qml.state()
        return circuit()

    # ------------------------------------------------------------------
    # Inner loop: angle optimisation
    # ------------------------------------------------------------------
    def _optimize_circuit(self) -> Tuple[float, np.ndarray]:
        """Optimise all rotation angles of the current circuit with PyTorch Adam.

        The circuit is simulated by a PennyLane QNode with the torch interface and
        differentiated with torch autograd. Stops after ``self.epochs`` iterations,
        when the infidelity drops below the threshold, or on a plateau
        (:meth:`check_early_stopping`, in which case the last step is rolled back).
        Updates ``self.params`` in place.

        Returns:
            Tuple[float, np.ndarray]: ``(best_dist, statevector)`` where

            * ``best_dist`` (float): best infidelity seen during the optimisation
              (1.0 if the circuit has no rotation gate yet);
            * ``statevector`` (np.ndarray, shape ``(2**n_qubits,)``): circuit output
              for the final ``self.params``.
        """
        if len(self.params) == 0:
            # no angles at all (only CNOTs so far): state is still |0...0>
            init_psi = np.zeros(2 ** self.n_qubits, dtype=np.complex128)
            init_psi[0] = 1.0
            return 1.0, init_psi

        best_dist = 1.0

        @qml.qnode(self.dev)
        def get_state_node(angles):
            """QNode applying ``self.current_gates`` with the given angles.

            Args:
                angles (torch.Tensor | np.ndarray): rotation angles, one per rotation gate.

            Returns:
                torch.Tensor | np.ndarray: output statevector, shape ``(2**n_qubits,)``
                (a torch tensor when ``angles`` is a torch tensor).
            """
            _queue_gates(self.current_gates, angles)
            return qml.state()

        def cost_fn(angles):
            """Cost minimised by Adam: the infidelity ``L = 1 - |<target|psi(angles)>|^2``.

            Args:
                angles (torch.Tensor): rotation angles, one per rotation gate (requires grad).

            Returns:
                torch.Tensor: scalar infidelity, differentiable w.r.t. ``angles``.
            """
            curr_psi = get_state_node(angles)
            inner_prod = qml.math.dot(qml.math.conj(self.target_psi), curr_psi)
            fidelity = qml.math.abs(inner_prod) ** 2
            return 1.0 - fidelity  # infidelity, L = 1 - F (Eq. 9)

        self.history = []
        self._es_best = np.inf
        self._es_stall_count = 0

        _limit_torch_threads_once()
        params_t = torch.tensor(np.array(self.params, dtype=float), requires_grad=True)
        optimizer = torch.optim.Adam([params_t], lr=self.lr)

        e = 0
        while e < self.epochs:
            prev_params_t = params_t.detach().clone()
            optimizer.zero_grad()
            cost = cost_fn(params_t)
            cost.backward()
            optimizer.step()

            dist_val = float(cost.detach())
            self.history.append(dist_val)
            if dist_val < best_dist:
                best_dist = dist_val
            if best_dist < self.threshold:
                break

            if self.check_early_stopping(e):
                # plateau: roll back the last step and stop
                with torch.no_grad():
                    params_t.copy_(prev_params_t)
                break
            e += 1

        # hand back a plain pnp array, same type self.params always has --
        # everything downstream (warmstart export, next step's pnp.append,
        # the final state evaluation below) is unaware torch was involved.
        self.params = pnp.array(params_t.detach().numpy(), requires_grad=True)

        # Final pass: statevector for the RL observation
        @qml.qnode(self.dev)
        def get_final_psi(angles):
            """QNode evaluating the circuit output for the optimised angles.

            Args:
                angles (pnp.ndarray): optimised rotation angles, one per rotation gate.

            Returns:
                np.ndarray: output statevector, shape ``(2**n_qubits,)``.
            """
            _queue_gates(self.current_gates, angles)
            return qml.state()

        final_psi = get_final_psi(self.params)

        return best_dist, np.array(final_psi)

    def check_early_stopping(self, epoch: int) -> bool:
        """Patience-based plateau detection for the inner Adam loop.

        Counts iterations since the loss last improved on the best value by a
        scale-relative margin (one order of magnitude below the current loss), and
        signals a plateau once that count reaches ``self.es_epochs``. Robust to noise
        around the loss floor: only a genuine improvement over the best value seen so
        far resets the counter. Reads ``self.history`` and updates ``self._es_best``
        and ``self._es_stall_count``.

        Args:
            epoch (int): current inner iteration (unused, kept for readability at the call site).

        Returns:
            bool: True if the optimisation has plateaued and should stop.
        """
        current = self.history[-1]
        threshold = 10 ** (math.floor(math.log10(max(current, 1e-300))) - 1)
        if current < self._es_best - threshold:
            self._es_best = current
            self._es_stall_count = 0
        else:
            self._es_stall_count += 1
        return self._es_stall_count >= self.es_epochs

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------
    def _gates_commute(self, g1: Gate, g2: Gate) -> bool:
        """Sufficient (not necessary) condition for two gates to commute.

        True if the gates act on disjoint wires, are CNOTs with the same control and
        different targets, or are same-axis rotations on the same qubit.

        Args:
            g1 (Gate): first gate dict (``gate``, ``wires``, ``is_rot``).
            g2 (Gate): second gate dict.

        Returns:
            bool: True if ``g1`` and ``g2`` are known to commute.
        """
        # 1. If they share no qubits, they commute
        if set(g1['wires']).isdisjoint(g2['wires']):
            return True

        # 2. CNOTs sharing the same control qubit commute with each other
        if (not g1['is_rot']) and (not g2['is_rot']):
            # wires is [control, target]
            if g1['wires'][0] == g2['wires'][0] and g1['wires'][1] != g2['wires'][1]:
                return True

        # 3. Rotations of the same type (e.g., Ry & Ry) on the same qubit commute
        if g1['is_rot'] and g2['is_rot']:
            if g1['gate'] == g2['gate'] and g1['wires'] == g2['wires']:
                return True

        return False

    def get_action_mask(self) -> np.ndarray:
        """Mask of the actions allowed in the current state.

        An action is forbidden if:

        * it is a CNOT whose control qubit is still in |0> (no-op);
        * it is a rotation on a qubit that already received 3 rotations in the
          trailing run of single-qubit gates (any SU(2) is reachable with 3);
        * an identical gate can be reached walking back through the history across
          gates it commutes with, i.e. the new gate would merge with it (same-axis
          rotation) or cancel it (same CNOT twice).

        Returns:
            np.ndarray: float32 array of shape ``(action_dim,)``, 1.0 = allowed, 0.0 = forbidden.
        """
        mask = np.ones(self.action_dim, dtype=np.float32)

        history = getattr(self, 'applied_action_indices', [])

        # rotations per qubit since the last entangling gate
        consecutive_rotations = [0] * self.n_qubits
        for past_action_idx in reversed(history):
            past_act = self.actions_map[past_action_idx]
            if past_act['is_rot']:
                consecutive_rotations[past_act['wires'][0]] += 1
            else:
                break

        for i, act in enumerate(self.actions_map):
            # CNOT with control still in |0> is useless
            if not act['is_rot']:
                control_qubit = act['wires'][0]
                if not self.qubit_active[control_qubit]:
                    mask[i] = 0.0
                    continue

            if act['is_rot']:
                target_wire = act['wires'][0]
                if consecutive_rotations[target_wire] >= 3:
                    mask[i] = 0.0

            is_canceling = False

            # Scan backward through the gate history of the current episode
            for past_action_idx in reversed(history):
                past_act = self.actions_map[past_action_idx]

                # Found the identical gate?
                if act['is_rot'] == past_act['is_rot'] and act['wires'] == past_act['wires']:
                    # For rotations, the rotation axis (gate name) must match too
                    if (not act['is_rot']) or (act['gate'] == past_act['gate']):
                        is_canceling = True
                        break

                # A non-commuting gate in between blocks the merge/cancellation
                if not self._gates_commute(act, past_act):
                    break

            if is_canceling:
                mask[i] = 0.0
        return mask

    def action_masks(self) -> np.ndarray:
        """Boolean action mask, the interface expected by sb3-contrib ``MaskablePPO``.

        Returns:
            np.ndarray: bool array of shape ``(action_dim,)``, True = allowed.
        """
        return self.get_action_mask().astype(bool)
