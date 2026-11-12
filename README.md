# Curriculum Learning-Based Reinforcement Learning Agent
for Approximate Quantum State Preparation

Code for the paper **"Curriculum Learning-Based Reinforcement Learning Agent
for Approximate Quantum State Preparation"** (M. Mordacci, M. Amoretti).

A PPO agent builds a circuit gate by gate, while PyTorch Adam tunes the rotation angles
after every placement. The target is reached within an infidelity of 10⁻²
with as few gates as possible. Larger systems are reached through a
**qubit-by-qubit curriculum** that warm-starts each stage from the circuit
*and* the network trained on the previous one. The framework is evaluated on
W, Dicke₂ (k = 2) and random states.

Built on [PennyLane](https://pennylane.ai), [Gymnasium](https://gymnasium.farama.org),
[Stable-Baselines3](https://stable-baselines3.readthedocs.io) and [PyTorch](https://https://pytorch.org/).
**All default values are the configuration used in the paper.**

---

## Contents

- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the paper](#reproducing-the-paper)
- [Output files](#output-files)
- [Configuration reference](#configuration-reference)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [Citation](#citation)

---

## How it works

### Two nested loops

| Loop | Who | Decides |
|---|---|---|
| **Outer** | PPO agent | *which* gate to append: `Rx`/`Ry`/`Rz` on qubit *q*, or `CNOT` on an ordered pair *(c, t)* |
| **Inner** | PyTorch Adam (lr 0.05, ≤ 200 iterations, early stopping) | the angles of **all** rotation gates of the circuit, minimising L = 1 − F |

At each RL step the chosen gate is appended (a new rotation angle starts from
a small random perturbation around 0), the inner loop re-optimises all angles
jointly, and the resulting distance drives the reward.

An episode starts from |0⟩^⊗n (or from the warm-start circuit, see below),
*terminates* when 1 − F < `--threshold` (10⁻²), and is *truncated* after
`--max_gates` gates.

### MDP

- **Actions** – `3n` single-qubit rotations + `n(n−1)` CNOTs. No angles.
- **Observation** – `[Re ψ, Im ψ, gate_count / max_gates, Re(ψ_t − ψ), Im(ψ_t − ψ)]`,
  length `4·2ⁿ + 1` (current state, normalised gate count, residual w.r.t. the target).
- **Action masking** – forbidden actions:
  - CNOTs whose control qubit is still in |0⟩;
  - gates that would merge with or cancel an identical earlier gate reachable
    through gates they commute with (same-axis rotation on the same qubit,
    same CNOT twice);
- **Networks** – actor and critic are two separate MLPs with three hidden
  layers of 256 units (ReLU by default, `--act_fun`).
- **Reward** (Eq. 10–11 of the paper), with D the distance after the step:

  r = λ [log(D<sub>t−1</sub> + ε) − log(D<sub>t</sub> + ε)] − p, &nbsp; λ = `--q_loss_mult` = 10, p = −`--q_penalty` = 1

  plus, with `--var_rew` (on):
  - a +20 milestone bonus the first time (within an episode) D drops below
    each order of magnitude 10⁻¹, 10⁻², … down to the threshold, included
    (at the default threshold: 10⁻¹ and 10⁻²);
  - on success, r_b = β exp(−0.5 · max(g − g*, −2)), β = `--q_exp` = 75, where
    g is the gate count and g* the best gate count found so far (10 for the
    first success).

### Curriculum and warm-starts

- **W / Dicke₂:** each stage's target is the closed-form k-qubit state.
- **Random states (top-down):** sample a random n-qubit state, then
  recursively extract the dominant Schmidt vector across the last qubit
  (n → n−1 → … → 2). Each stage's target is the best rank-one (k−1)-qubit
  factor of the next stage's target.

Each stage is warm-started from the previous one in two ways:

1. **Circuit warm-start** – the best circuit of stage k−1 is replayed at the
   start of *every* episode of stage k (the new qubit starts in |0⟩); the
   agent learns the extra gates needed.
2. **Network warm-start** – every actor/critic tensor whose name and shape
   match is copied from the previous stage's best model. The hidden layers
   transfer; the input layer and the action head change size with *n* and are
   freshly initialised.

---

## Repository structure

```
.
├── prepare_curriculum.py   # step 1: generate stage targets + manifest.json
├── trainSB3.py             # step 2: train one curriculum stage
├── requirements.txt
└── qcsynth/
    ├── env.py              # QuantumGymEnv: MDP, inner PyTorch-Adam loop, reward, action masking
    ├── states.py           # W, Dicke, random states + Schmidt decomposition
    ├── networks.py         # actor/critic MLPs (3 × 256) + cross-stage weight transfer
    ├── callbacks.py        # logging, best-circuit tracking, checkpoints, early stopping
    ├── warmstart.py        # warm-start JSON export
    ├── manifest.py         # --manifest resolution (one job per stage)
    ├── env_factory.py      # CLI args -> environment constructor for SubprocVecEnv
    └── cli.py              # all trainSB3.py options and their (paper) defaults
```

---

## Installation

Tested with Python 3.13 and the package versions listed in `requirements.txt`.

```bash
git clone https://github.com/qslab-unipr/CL-PPO_QSP.git
cd CL-PPO_QSP

conda create -n env_name python=3.13 -y
conda activate env_name
pip install -r requirements.txt
```

Everything runs on CPU: the circuits are simulated with PennyLane's
`default.qubit` and the networks are small. On a CPU-only machine (e.g. a
cluster node) you can install the smaller CPU build of PyTorch before the
other requirements:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Reproducing the paper

### 1. Prepare the curriculum

```bash
python prepare_curriculum.py --n_qubits 13 --start_qubit 3 --state_type W  --base_path experiments
python prepare_curriculum.py --n_qubits 10 --start_qubit 4 --state_type dicke --base_path experiments
python prepare_curriculum.py --n_qubits 7  --state_type random --seed 0 --base_path experiments
```

This creates, e.g. for W:

```
experiments/W_n13/
├── manifest.json            # one entry per stage (target, run_name, warm-start path)
├── targets/target_2.npy … target_13.npy
├── models/                  # --saving_path of every stage
└── logs/                    # convenience folder, e.g. for SLURM output
```

| Option | Default | Description |
|---|---|---|
| `--n_qubits` | *(required)* | Final number of qubits |
| `--state_type` | `random` | `W`, `dicke` or `random` |
| `--base_path` | `/experiments` | Root output folder — **set this explicitly** |
| `--experiment_name` | `<type>_n<N>[_seed<S>]` | Sub-folder name |
| `--seed` | auto | Seed for `random` (if omitted, one is generated and saved in the manifest) |
| `--start_qubits` | `2` | First stage (trained from scratch, no warm-start) |
| `--run_label` | – | Prefix for run names (`<label>_stage<k>`) |

For random targets the script prints, at every extraction step, the fidelity
ceiling λ₀ reached by the replayed (k−1)-qubit circuit against the k-qubit
target before any new gate is added.

### 2. Train the stages, in order

```bash
M=experiments/W_n13/manifest.json
python trainSB3.py --manifest $M --n_qubits 2  --max_gates 20 --num_envs 1
python trainSB3.py --manifest $M --n_qubits 3  --max_gates 20
python trainSB3.py --manifest $M --n_qubits 4  --max_gates 30
# ...
```

All other hyper-parameters already default to the paper values. With
`--manifest`, `trainSB3.py`:

- sets `--state custom`, the target file, the warm-start file, `--run_name`
  and `--saving_path` automatically;
- **refuses to start** (exit code 1) if the previous stage's
  `<run_name>_warmstart.json` does not exist yet;
- retrains a stage even if it already has a `<run_name>_final.zip`
  (`--force` is on by default; use `--no_force` to skip finished stages);
- stops a stage early once it has converged (see [early stopping](#early-stopping-on-by-default));
- loads the previous stage's `best_model.zip` for the network warm-start.

`--max_gates` counts the replayed warm-start circuit too, so it grows with
*n*. Values used in the paper (16 parallel environments, 1 for two qubits):

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **W** | – | 20 | 30 | 40 | 50 | 70 | 70 | 70 | 70 | 70 | 80 | 90 |
| **Dicke₂** | – | – | 30 | 40 | 70 | 80 | 140 | 150 | 200 | – | – | – |
| **Random** | 20 | 40 | 60 | 80 | 150 | 350 | – | – | – | – | – | – |

---

## Output files

Everything is written to the manifest's `models/` folder (`--saving_path`):

| File | When | Content |
|---|---|---|
| `<run>_warmstart.json` | every new best circuit, every checkpoint, on SIGTERM, at the end | best circuit (`gates`, `params`, `dist`, `model_path`) — the input of the next stage |
| `<run>_best_model.zip` | on new best circuit (throttled by `--best_model_min_interval`) | policy that found it — source of the network warm-start |
| `<run>_step<N>.zip` | every `--save_every` timesteps | periodic checkpoint |
| `<run>_final.zip` | end of training | final model |
| `<run>_interrupted.zip` | on SIGTERM | model at kill time |
| `<run>_vecnormalize.pkl` | end of training | reward-normalisation statistics |
| `runs/<run>/` | continuously | TensorBoard logs |

The best circuit is the one with the fewest gates, ties broken by lower
distance (truncated episodes count too, so before the first success it is the
best partial circuit). It is also printed at every improvement:

```
New shortest circuit: 5 gates, dist=3.2e-03
  Ry(wire=0, angle=1.910633 rad  [109.47°])
  CNOT(ctrl=0, tgt=1)
  ...
```

---

## Configuration reference

`python trainSB3.py --help` lists every option with its default. Options that
are on by default have a `--no_<option>` switch.

### Environment and reward

| Option | Default | Description |
|---|---|---|
| `--n_qubits` | 2 | Number of qubits (stage selector with `--manifest`) |
| `--state` | `custom` | `W`, `dicke` or `custom` (forced to `custom` by `--manifest`) |
| `--max_gates` | 30 | Gates per episode, warm-start included (see table above) |
| `--threshold` | 1e-2 | Success when 1 − F < threshold |
| `--q_loss_mult` | 10 | λ, weight of the log-distance improvement |
| `--q_penalty` | −1 | −p, per-step penalty |
| `--var_rew` / `--no_var_rew` | on | Milestone + exponential record bonus |
| `--q_exp` | 75 | β, scale of the record bonus |
| `--var_pen` | off | Multiply the step penalty by the gate count |
| `--q_clip` | off | Clip the step reward to [−10, 10] |

### Inner loop

| Option | Default | Description |
|---|---|---|
| `--q_lr` | 0.05 | Adam learning rate |
| `--q_epochs` | 200 | Max iterations per RL step |
| `--es_epochs` | 20 | Inner early-stopping patience |

### PPO

| Option | Default | Description |
|---|---|---|
| `--num_envs` | 16 | Parallel environments (paper: 1 for two qubits) |
| `--steps` | 128 | Rollout length per env (batch size = steps · num_envs / 4) |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 10 | PPO epochs per update |
| `--ent_coef` | 0.01 | Entropy coefficient |
| `--gamma` | 0.99 | Discount |
| `--eps_clip` | 0.2 | PPO clip range |
| `--act_fun` | `relu` | `relu` or `tanh` |
| `--norm_reward` / `--no_norm_reward` | on | `VecNormalize` reward normalisation (observations are not normalised) |
| `--clip_reward` | 10 | Clip of the normalised reward |
| `--network_warmstart` / `--no_network_warmstart` | on | Cross-stage weight transfer |
| `--total_steps` | 1e9 | Timestep budget (in practice ended by early stopping or wall-clock) |
| `--save_every` | 50 000 | Checkpoint period (timesteps) |

### Early stopping (on by default)

All values in **timesteps**; the success rate is computed over the last 100
episodes. Nothing is checked before `--early_stop_min_steps` (50 000). Then
training stops when any trigger fires:

| Trigger | Condition |
|---|---|
| soft (paper rule) | no strictly shorter circuit for `--early_stop_patience_steps` (10 000) **and** success rate ≥ `--early_stop_min_success_rate` (0.98) |
| hard | no strictly shorter circuit for `--early_stop_hard_patience_steps` (450 000), regardless of success rate |
| plateau | the success rate improved by less than `--early_stop_plateau_min_delta` (0.02) over `--early_stop_plateau_window_steps` (220 000) |

A same-length circuit with better fidelity updates the warm-start but does
not reset the patience. Early stopping ends the run normally (final model and
warm-start are saved); otherwise the run is ended by the job wall-clock (3
days in the paper), with the state saved on `SIGTERM`. Disable it with
`--no_early_stop`.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir experiments/W_n13/models/runs
```

| Tag | Meaning |
|---|---|
| `quantum/infidelity`, `quantum/log10_dist` | final infidelity 1 − F of the episode (and its log10) |
| `quantum/best_dist_ever` | best distance so far |
| `quantum/gate_count`, `quantum/gates_on_success` | circuit length (all / successful episodes) |
| `quantum/success_rate_100ep`, `quantum/success_rate_total` | success rates |
| `quantum/ep_reward`, `quantum/reward_per_gate` | episode return (before `VecNormalize`) |

---

## Citation

```bibtex
@article{mordacci2026curriculum,
  title   = {Curriculum Learning-Based Reinforcement Learning Agent for Approximate Quantum State Preparation},
  author  = {Mordacci, Marco and Amoretti, Michele},
  year    = {2026}
}
```
