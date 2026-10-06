"""Evaluation protocols that separate objective from policy and exposure (revision step 6).

The standard readout drives every arm with its OWN actor head. For the survival arm that is
the trained policy; for the untrained and predictor arms it is an actor head that never
received an actor loss, so their evaluation trajectories come from an arbitrary
deterministic policy that differs per arm. A difference in decodability between arms can
then come from the objective, from the policy that generated the evaluation data, or from
what each arm was exposed to in training. Three protocols pull these apart.

  scripted   every arm and both pools follow the SAME scripted action sequence at episode
             index i (the predictor's own training policy, `experiment_b.scripted_policy`,
             with rng seed policy_seed_base + i). Physical trajectories then differ across
             pools only through the dynamics, and across arms not at all.
  replay     recorded trajectories (raw observations and the actions taken) of one agent
             are fed open loop into another arm's trunk, so every arm is read on identical
             input sequences. Typically the survival agent's own evaluation episodes.
  exposure   `train_predictor_on_logged` trains the prediction objective on trajectories
             logged from another agent (for example the trained survival policy in its
             training world), matching data exposure while changing only the objective.

Each arm keeps its own frozen normalizer; nothing here updates one.
"""

from __future__ import annotations

import numpy as np
import torch

from .agent_ac import RecurrentActorCritic
from .experiment_a import grouped_auroc
from .experiment_b import episode_features, scripted_policy
from .experiment_b2 import RunningNorm, _seeds, make_world
from .stats import auroc_ci

POLICY_SEED_BASE = 600_000   # disjoint from every world seed base in the pipeline


def collect_pool_scripted(agent, norm, params, drift_sigma, n_eps, steps, device, seed_base,
                          ray_steps, policy_seed_base: int = POLICY_SEED_BASE):
    """Like collect_pool, but actions come from the scripted policy with rng seed
    policy_seed_base + i, not from the agent. The agent's GRU receives each observation and
    the scripted action as its previous action, exactly as in predictor training. Returns
    (H (k, steps, hidden), kept indices) for full-length survivors."""
    Hs, kept = [], []
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        w.reset(_seeds(seed_base + i))
        rng = np.random.default_rng(policy_seed_base + i)
        h = agent.initial_state(1, device)
        prev = torch.zeros(1, agent.act_dim, device=device)
        obs = w.observe().astype(np.float64)
        row, died = [], False
        with torch.no_grad():
            for _ in range(steps):
                o = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
                h = agent.step_state(o, prev, h)
                row.append(h[0].cpu().numpy())
                a = scripted_policy(rng)
                r = w.step(a)
                prev = torch.as_tensor(a[None], dtype=torch.float32, device=device)
                obs = r.obs.astype(np.float64)
                if r.terminated:
                    died = True
                    break
        if not died and len(row) == steps:
            Hs.append(np.asarray(row, np.float32))
            kept.append(i)
    H = np.stack(Hs) if Hs else np.zeros((0, steps, agent.hidden), np.float32)
    return H, np.asarray(kept, dtype=int)


def record_trajectories(agent, norm, params, drift_sigma, n_eps, steps, device, seed_base,
                        ray_steps, deterministic: bool = True):
    """Run `agent` with its own policy and record, per full-length episode, the RAW
    observation it saw at each step and the env action it took. Returns (obs (k, steps, O),
    act (k, steps, A), kept indices)."""
    O, A, K = [], [], []
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        w.reset(_seeds(seed_base + i))
        h = agent.initial_state(1, device)
        prev = torch.zeros(1, agent.act_dim, device=device)
        obs = w.observe().astype(np.float64)
        orow, arow, died = [], [], False
        for _ in range(steps):
            o = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
            _, env_act, _, _, h = agent.act(o, prev, h, deterministic=deterministic)
            a = env_act[0].detach().cpu().numpy().astype(np.float32)
            orow.append(obs.astype(np.float32))
            arow.append(a)
            r = w.step(a)
            prev = env_act
            obs = r.obs.astype(np.float64)
            if r.terminated:
                died = True
                break
        if not died and len(orow) == steps:
            O.append(np.stack(orow))
            A.append(np.stack(arow))
            K.append(i)
    if not O:
        return (np.zeros((0, steps, agent.obs_dim), np.float32),
                np.zeros((0, steps, agent.act_dim), np.float32), np.zeros(0, int))
    return np.stack(O), np.stack(A), np.asarray(K, dtype=int)


@torch.no_grad()
def replay_states(agent, norm, obs_raw: np.ndarray, acts: np.ndarray, device="cpu") -> np.ndarray:
    """Recurrent states of `agent` on recorded trajectories, open loop: step t receives the
    recorded observation (normalized with the agent's own frozen normalizer) and the
    recorded action of step t-1 (zeros at t = 0). (k, steps, hidden)."""
    k, T, _ = obs_raw.shape
    if k == 0:
        return np.zeros((0, T, agent.hidden), np.float32)
    x = torch.as_tensor(norm(obs_raw.astype(np.float64)), dtype=torch.float32, device=device)
    a_in = np.zeros_like(acts)
    a_in[:, 1:] = acts[:, :-1]
    return agent.states_for_probe(x, torch.as_tensor(a_in, device=device)).cpu().numpy()


def readout_from_states(Ha: np.ndarray, Hs: np.ndarray, seed: int = 0,
                        groups: np.ndarray | None = None) -> dict:
    """The standard pooled probe ([mean h, final h], grouped CV) on given state pools."""
    out = {"n_auth": int(len(Ha)), "n_surr": int(len(Hs)), "target": float("nan"),
           "oof_ci95": [float("nan"), float("nan")]}
    if len(Ha) < 5 or len(Hs) < 5:
        return out
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    auc, yv, pv = grouped_auroc(X, y, g, return_oof=True)
    out["target"] = auc
    if yv.size:
        out["oof_ci95"] = list(auroc_ci(yv, pv, seed=seed))
    return out


def train_predictor_on_logged(batches: list, *, embed: int = 64, hidden: int = 96,
                              lr: float = 1e-3, seed: int = 0, device: str = "cpu"):
    """The prediction objective of `train_predictor_only`, trained on LOGGED batches instead
    of scripted-policy rollouts: one optimizer update per logged batch, in the logged order,
    so the update count, batch size, and episodes match the logging agent's training exactly
    (`train_actor_critic(..., log_batches=...)`). Each batch is a dict of raw observations
    (B, T, O), env actions (B, T, A), and mask (B, T). Same trunk, loss, optimizer, and
    gradient clip as `train_predictor_only`. The normalizer is updated with each batch's
    valid observations before that batch is used and frozen at the end."""
    O = batches[0]["obs_raw"].shape[-1]
    A = batches[0]["env_act"].shape[-1]
    torch.manual_seed(seed)
    agent = RecurrentActorCritic(O, A, embed, hidden, True).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    norm = RunningNorm(O)
    for b in batches:
        m = b["mask"] > 0.5
        norm.update(b["obs_raw"][m].astype(np.float64))
        x = torch.as_tensor(norm(b["obs_raw"].astype(np.float64)) * m[..., None],
                            dtype=torch.float32, device=device)
        a_env = torch.as_tensor(b["env_act"], device=device)
        a_in = torch.zeros_like(a_env)
        a_in[:, 1:] = a_env[:, :-1]
        mask = torch.as_tensor(b["mask"], device=device)
        loss, _ = agent.world_model_loss(x, a_in, a_env, mask,
                                         agent.initial_state(x.shape[0], device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
    return agent.train(False), norm.freeze()
