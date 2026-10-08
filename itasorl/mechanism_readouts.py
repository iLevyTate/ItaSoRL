"""Mechanism readouts of the goal-and-stakes spec
(docs/specs/2026-10-07-goal-and-stakes-design.md, section "Mechanism readouts").

  rollout_behavior               own-policy episodes with an optional constant state nudge;
                                 returns behavior means, intake halves, states, prediction error
  aggregate_behavior             per-episode behavior row from raw per-step rows
  probe_direction                the registered pooled probe's unit direction in state space
  sham_direction                 a fixed-RNG unit direction orthogonal to it
  gap_closed                     fraction of the authentic-to-surrogate behavior gap a nudge closes
  behavior_scale / behavior_difference   standardized between-world behavior comparison
  surprise_summaries / surprise_auroc / direction_error_correlation   the surprise channel
  adaptation                     within-lifetime intake recovery
  scripted_observation_streams   agent-free observation streams under the scripted policy

Everything is deterministic given seeds. Behavior columns are BEHAVIOR_NAMES; the first
five are the spec's measures, the last two the action-variability extras of readout 7.
"""

from __future__ import annotations

import warnings

import numpy as np

from itasorl.experiment_a import grouped_auroc
from itasorl.experiment_b import episode_features, scripted_policy

BEHAVIOR_NAMES = ("speed", "abs_turn", "thrust", "intake_rate", "near_food",
                  "std_thrust", "std_turn")
N_MEASURES = 5   # the spec's behavior vector; the two std columns are readout-7 extras
ACT_DIM = 5      # env action: thrust, turn, eat, drink, rest (scripted_policy's length)
MIN_AUROC_ROWS = 5   # surprise_auroc needs at least this many finite rows per class


def aggregate_behavior(R: np.ndarray) -> np.ndarray:
    """Per-episode behavior row (7,) from raw per-step rows R (steps, 6):
    [speed, |turn|, thrust, intake, near_food, signed turn]. The first five columns are
    averaged (the spec's measures); the extras are the within-episode std of thrust and of
    the SIGNED turn action, so alternating left/right at constant magnitude counts as
    variable."""
    R = np.asarray(R, dtype=np.float64)
    return np.r_[R[:, :N_MEASURES].mean(0), R[:, 2].std(), R[:, 5].std()]


def rollout_behavior(agent, norm, params, drift_sigma: float, *, n_eps: int, steps: int,
                     seed_base: int, ray_steps: int, nudge: np.ndarray | None = None,
                     device: str = "cpu") -> dict:
    """Deterministic own-policy episodes of exactly `steps` steps (an episode that dies
    earlier is dropped, as collect_pool does). `nudge` (hidden,) is added to the hidden
    state after every GRU update and before the policy head, so the action at step t, the
    recorded state, the decoder input, and the next GRU step all read the nudged state.
    Returns B (k, 7) behavior per episode, halves (k, 2) intake rate in the first and second
    half, H (k, steps, hidden) the (nudged) states, E (k, steps-1) squared next-observation
    prediction error of the decoder (nan if the agent has none), kept episode indices.
    Callers comparing variants (nudged vs unnudged, authentic vs surrogate) should intersect
    `kept` across them before taking means, so survivor selection cannot masquerade as a
    behavior shift."""
    import torch

    from itasorl.experiment_b2 import _seeds, make_world

    A = agent.act_dim
    nudge_t = None if nudge is None else torch.as_tensor(np.asarray(nudge, np.float32),
                                                        device=device)[None, :]
    B, halves, Hs, Es, kept = [], [], [], [], []
    half = steps // 2
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        obs = w.reset(_seeds(seed_base + i)).obs.astype(np.float64)
        h = agent.initial_state(1, device)
        prev = torch.zeros(1, A, device=device)
        rows, hs, es, intake = [], [], [], []
        alive = True
        for t in range(steps):
            obs_t = torch.as_tensor(norm(obs[None]), dtype=torch.float32, device=device)
            with torch.no_grad():
                h = agent.step_state(obs_t, prev, h)
                if nudge_t is not None:
                    h = h + nudge_t
                _, env_act, _, _ = agent.act_from_state(h, deterministic=True)
            a = env_act[0].detach().cpu().numpy().astype(np.float32)
            r = w.step(a)
            if agent.world_model and t < steps - 1 and not r.terminated:
                with torch.no_grad():
                    pred = agent.predict_next(h[:, None, :], env_act[:, None, :])[0, 0]
                nxt = torch.as_tensor(norm(r.obs[None]), dtype=torch.float32, device=device)[0]
                es.append(float(((pred - nxt) ** 2).mean()))
            d2 = float(np.min(np.sum((w.pellets - w.pos) ** 2, axis=1)))
            rows.append([float(np.linalg.norm(w.vel)), abs(float(a[1])), float(a[0]),
                         float(r.info.get("intake", 0.0)), float(d2 < w.reach ** 2),
                         float(a[1])])
            intake.append(float(r.info.get("intake", 0.0)))
            hs.append(h[0].detach().cpu().numpy().copy())
            prev = env_act
            obs = r.obs.astype(np.float64)
            if r.terminated:
                alive = False
                break
        if not alive:
            continue
        B.append(aggregate_behavior(np.asarray(rows)))
        halves.append([float(np.mean(intake[:half])), float(np.mean(intake[half:]))])
        Hs.append(np.stack(hs))
        Es.append(np.asarray(es, np.float32) if es else np.full(steps - 1, np.nan, np.float32))
        kept.append(i)
    hid = agent.hidden
    return {"B": np.asarray(B).reshape(-1, len(BEHAVIOR_NAMES)),
            "halves": np.asarray(halves).reshape(-1, 2),
            "H": np.stack(Hs) if Hs else np.zeros((0, steps, hid), np.float32),
            "E": np.stack(Es) if Es else np.zeros((0, steps - 1), np.float32),
            "kept": np.asarray(kept, int)}


def probe_direction(Ha: np.ndarray, Hs: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the registered pooled probe (StandardScaler + LogisticRegression on
    [mean h, final h]) on every episode and return (u, s): u the unit direction in RAW state
    space along which a constant shift most increases the surrogate score (mean-block plus
    final-block coefficients, each divided by its scaler scale), s the standard deviation of
    all pooled per-step states projected onto u. A degenerate probe (zero coefficient
    vector, e.g. identical pools) returns u = zeros, s = nan, and warns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hid = Ha.shape[-1]
    X = episode_features(np.concatenate([Ha, Hs]))
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    coef = clf[-1].coef_[0] / clf[0].scale_
    w = coef[:hid] + coef[hid:]
    if np.linalg.norm(w) < 1e-9:
        warnings.warn("degenerate probe direction", stacklevel=2)
        return np.zeros(hid, np.float64), float("nan")
    u = w / np.linalg.norm(w)
    proj = np.concatenate([Ha, Hs]).reshape(-1, hid) @ u
    return u.astype(np.float64), float(proj.std())


def sham_direction(u: np.ndarray, seed: int) -> np.ndarray:
    """A unit direction orthogonal to u, drawn once per seed from a fixed RNG (the sham)."""
    rng = np.random.default_rng(1_000_003 + seed)
    v = rng.normal(size=u.shape)
    v = v - (v @ u) * u
    return v / np.linalg.norm(v)


def behavior_scale(b_auth: np.ndarray, b_surr: np.ndarray) -> np.ndarray:
    """Pooled across-world per-column standard deviation (the spec's standardization);
    all-NaN when both pools are empty."""
    both = np.concatenate([b_auth, b_surr])
    if len(both) == 0:
        return np.full(both.shape[1:], np.nan)
    return both.std(0) + 1e-8


def gap_closed(b_auth: np.ndarray, b_surr: np.ndarray, b_nudged: np.ndarray,
               scale: np.ndarray) -> tuple[float, float]:
    """(score, gap): score is the fraction of the standardized authentic-to-surrogate
    behavior gap (first N_MEASURES columns) that the nudged behavior closes; gap is the
    standardized gap norm (the spec's informativeness threshold applies to it). An empty
    pool yields a NaN gap; a zero or NaN gap yields a NaN score."""
    m = N_MEASURES
    if min(len(b_auth), len(b_surr), len(b_nudged)) == 0:
        return float("nan"), float("nan")
    ga = (b_surr[:, :m].mean(0) - b_auth[:, :m].mean(0)) / scale[:m]
    gn = (b_nudged[:, :m].mean(0) - b_auth[:, :m].mean(0)) / scale[:m]
    gap = float(np.linalg.norm(ga))
    if not np.isfinite(gap) or gap == 0.0:
        return float("nan"), float(gap)
    return float(gn @ ga / gap ** 2), gap


def behavior_difference(b_auth: np.ndarray, b_surr: np.ndarray, scale: np.ndarray) -> dict:
    """Standardized surrogate-minus-authentic mean of every BEHAVIOR_NAMES column
    (readout 7); NaNs when either pool is empty."""
    if len(b_auth) == 0 or len(b_surr) == 0:
        return {name: float("nan") for name in BEHAVIOR_NAMES}
    d = (b_surr.mean(0) - b_auth.mean(0)) / scale
    return {name: float(v) for name, v in zip(BEHAVIOR_NAMES, d)}


def surprise_summaries(E: np.ndarray) -> np.ndarray:
    """Per-episode [mean, max, slope] of the prediction-error trace. A row with any
    non-finite value (an agent without a decoder) yields NaNs rather than a polyfit error."""
    t = np.arange(E.shape[1])
    slope = np.array([np.polyfit(t, e, 1)[0] if np.all(np.isfinite(e)) else np.nan for e in E])
    return np.c_[E.mean(1), E.max(1), slope]


def surprise_auroc(Ea: np.ndarray, Es: np.ndarray) -> float:
    """Readout 8(a): grouped-CV AUROC of the registered probe on per-episode surprise
    summaries, authentic (0) vs surrogate (1). Rows with non-finite summaries are dropped;
    NaN if fewer than MIN_AUROC_ROWS finite rows remain in either class."""
    X = np.concatenate([surprise_summaries(Ea), surprise_summaries(Es)])
    y = np.r_[np.zeros(len(Ea)), np.ones(len(Es))].astype(int)
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if (y == 0).sum() < MIN_AUROC_ROWS or (y == 1).sum() < MIN_AUROC_ROWS:
        return float("nan")
    return float(grouped_auroc(X, y, np.arange(len(y))))


def direction_error_correlation(H: np.ndarray, E: np.ndarray, u: np.ndarray) -> float:
    """Pearson r over all steps between the projection of h_t onto u and e_t (t < steps-1)."""
    proj = (H[:, : E.shape[1], :] @ u).ravel()
    e = E.ravel()
    ok = np.isfinite(e)
    if ok.sum() < 3 or proj[ok].std() == 0 or e[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(proj[ok], e[ok])[0, 1])


def adaptation(halves_auth: np.ndarray, halves_surr: np.ndarray) -> dict:
    """Readout 9: gap = authentic minus surrogate intake rate per half; adaptation =
    first-half gap minus second-half gap, positive when the gap narrows. NaNs when either
    input has zero rows."""
    if len(halves_auth) == 0 or len(halves_surr) == 0:
        return {"gap_first": float("nan"), "gap_second": float("nan"),
                "adaptation": float("nan")}
    g1 = float(halves_auth[:, 0].mean() - halves_surr[:, 0].mean())
    g2 = float(halves_auth[:, 1].mean() - halves_surr[:, 1].mean())
    return {"gap_first": g1, "gap_second": g2, "adaptation": g1 - g2}


def scripted_observation_streams(norm, params, drift_sigma: float, *, n_eps: int, steps: int,
                                 seed_base: int, ray_steps: int,
                                 policy_seed_base: int = 600_000) -> tuple[np.ndarray, np.ndarray]:
    """The agent-free observation stream for the scripted-walk decoder (the cell FINDINGS
    17.8 left missing): walks under the scripted policy, returning normalized observations
    (k, steps, O) and the previous action (k, steps, ACT_DIM) for full-length survivors."""
    from itasorl.experiment_b2 import _seeds, make_world

    O, A = [], []
    for i in range(n_eps):
        w = make_world(params, drift_sigma, ray_steps)
        obs = w.reset(_seeds(seed_base + i)).obs.astype(np.float64)
        rng = np.random.default_rng(policy_seed_base + i)
        prev = np.zeros(ACT_DIM, np.float32)
        o_rows, a_rows, alive = [], [], True
        for _ in range(steps):
            o_rows.append(norm(obs[None])[0].astype(np.float32))
            a_rows.append(prev.copy())
            a = scripted_policy(rng)
            r = w.step(a)
            prev = a
            obs = r.obs.astype(np.float64)
            if r.terminated:
                alive = False
                break
        if alive:
            O.append(np.stack(o_rows))
            A.append(np.stack(a_rows))
    return (np.stack(O) if O else np.zeros((0, steps, norm.mean.shape[0]), np.float32),
            np.stack(A) if A else np.zeros((0, steps, ACT_DIM), np.float32))
