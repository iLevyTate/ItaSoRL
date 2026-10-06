# What each agent experienced: training and evaluation, arm by arm

Revision step 6 (`docs/REVISION_2026-10.md`). This table describes the code as it runs
(`scripts/run_expB2.py`, `itasorl/experiment_b2.py`, `itasorl/eval_protocols.py`,
`scripts/run_expB_full.py`, `itasorl/experiment_c.py`). Where an earlier description says an
agent "lives in both worlds", this table is the correction.

## How a B-v2 / L3 cell is organized

Each (drift, seed) cell trains its agents in **one** condition and evaluates them on **two**.
A drift 0.45 cell trains every arm in the surrogate world only (the learned velocity law
`G_motion`); a drift 0 cell trains every arm in the authentic world only, and serves as the
L0 control. After training, each agent is frozen and scored on an authentic pool (world seeds
800000 + i, drift 0) and a pool at the cell's drift (850000 + i). No training run mixes the
two worlds, and no agent is trained on the authentic world and then tested on the surrogate,
except through the cross-evaluation return (`xeval`), which reports return only. A mixture
design would be a new experiment and is not what any published run did.

Nothing in an observation names the world. The observation does carry the consequences of the
dynamics (velocity in interoception and radial velocity in every vision ray), which is what
any detector reads.

## The arms

| Arm | Training world | Training policy | Evaluation policy (standard readout) | Objective | Optimization | Normalizer |
|---|---|---|---|---|---|---|
| `untrained` | none (random init) | none | its own **untrained** actor head, deterministic | none | none | fitted on 8 episodes x 40 steps of its own stochastic untrained policy at the cell's drift, then frozen |
| `predictor` | the cell's world (surrogate at drift 0.45) | fixed scripted policy (`experiment_b.scripted_policy`: thrust U(0, 0.6), turn U(-1, 1), eat with p 0.3) | its own actor head, **which never received an actor loss**, deterministic | next-observation prediction (MSE of the decoder from (h_t, a_t) to o_{t+1}) | Adam lr **1e-3**, 300 updates, 16 episodes per update, up to 80 steps each (at most 384,000 env steps), grad clip 1.0 | online on its own scripted-policy training observations, frozen after training |
| `survival` (published) | the cell's world | its own stochastic policy, learned | its own trained actor head, deterministic (mean action, binaries at p > 0.5) | A2C policy gradient + 0.5 x value loss - 0.01 x entropy + **1.0 x next-observation decoder loss**; GAE gamma 0.99, lambda 0.95; potential-based shaping on the training reward only | Adam lr 3e-4, 300 updates, 16 episodes per update, up to 80 steps each, one gradient step per update, grad clip 1.0 | online on its own training observations, frozen after training |
| `survival`, no auxiliary (`--no-world-model`) | the cell's world | learned | own trained actor head | as above without the decoder term | as above | as above |
| `survival`, sysid ceiling (`--sysid-aux`) | the cell's world | learned | own trained actor head | as above + 1.0 x MSE of a linear head from h_t to the drag drift (**supervised on the world variable**; a ceiling control, never a headline) | as above | as above |
| `predictor_logged` (new, step 6) | the survival agent's own training trajectories | none of its own: one update per logged survival batch, in order | its own actor head (no actor loss), and the shared protocols below | next-observation prediction | Adam lr 1e-3, 300 updates on exactly the survival agent's 300 batches | updated with each logged batch before use, frozen at the end |
| Experiment B predictor (`agent.py`) | a fixed **mixture**: 110 authentic + 110 surrogate scripted episodes | scripted | not applicable: probed on the same episodes it trained on | next-observation prediction | Adam lr 1e-3, 8 epochs, batch 16 | fitted on the training episodes |
| Experiment C population | mixed lifetimes: each policy lives authentic and surrogate episodes | each policy's own, deterministic | the population's policies, common-garden panel | selection on pooled foraging return; no gradient | 30 generations, 48 policies, mutation sigma 0.03 | per-policy, never updated mid-evaluation |

Reward shaping (survival arms): `shaping_coef` is **1.0** in `scripts/run_expB2.py` (the
`train_actor_critic` default of 0.5 is not what the runs used). The potential is minus the
distance to the nearest available pellet, it is set to 0 at a terminal step, and the shaped
term is `coef * (gamma * Phi(s') - Phi(s))`. Engagement and every reported return use the true
reward only. The rollout cutoff of 80 steps is a sampling truncation: from the 2026-10
correction on, a truncated episode bootstraps from the critic value of its successor state
(`docs/CORRECTIONS.md`).

## Why the predictor needs its own evaluation protocols

Under the standard readout the predictor is scored on trajectories its untrained actor head
generates, a policy it never learned and never saw in training (it trained on scripted
actions), with a normalizer fitted to scripted-policy observations. A low predictor target can
therefore reflect a distribution shift in its inputs as well as what its objective made it
represent. `scripts/run_policy_controlled_readouts.py` scores every arm three ways and
reports the three separately:

| Protocol | Who drives the evaluation episodes | What it holds fixed |
|---|---|---|
| `own` | each arm's own actor head (the standard readout) | nothing; reported as before |
| `scripted` | the same scripted action sequence for every arm and both pools (the predictor's training policy) | the action sequence; trajectories then differ across pools only through the dynamics, and across arms not at all |
| `replay` | the survival agent's own evaluation episodes, fed open loop to every arm | the input sequence every trunk reads |

The `predictor_logged` arm separates objective from data exposure: it is trained with the
prediction objective on exactly the batches the survival agent trained on (the survival agent
is retrained with logging, and the run stops unless the retrained agent is bit-identical to
the saved one). A survival-versus-predictor difference that persists against
`predictor_logged` under the `scripted` and `replay` protocols is a difference of objective at
matched exposure and matched evaluation input. Without that comparison, a survival-versus-
predictor gap is a gap between two training regimes that differ in objective, policy, and
data at once, and is described that way.
