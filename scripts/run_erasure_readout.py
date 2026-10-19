"""The removal test: erase the decoded subspace while the agent acts.

Frozen design: docs/specs/2026-10-15-online-erasure-design.md (with its pre-code amendment:
rank-two erasure). Readout-only.

    python scripts/run_erasure_readout.py --agents-dir fullruns/corrected_l3_h8_wm/agents \\
        --out artifacts/mechanism/erasure/corrected_l3_h8_wm.json

Full-rank erasure (closing item X10, E20 with E21 folded in; post hoc, descriptive):
`--max-rank K` (default 2), `--refit-stop A` (default none) and `--swap` (default off). With any
of them set, each cell first runs the rank-two readout unchanged (its values are the integrity
gate against --erasure-ref, within 0.01), then grows the erased subspace on the stored pools:
the registered probe is refit on the erased pools, each refit direction is projected onto the
complement of the erased subspace before it is added (a direction with almost nothing left
outside it is dropped, so the rank reads cleanly), and the subspace is extended. The stop
statistic is grown in fold: for each outer fold of the explicit twin-grouped partition the
subspace (from the rank-two block onward) is grown on that fold's training rows only, and at
each rank a probe fit on the erased training rows scores the erased test rows; the
out-of-fold scores are pooled across folds per rank (`refit_curve_infold`). k is the smallest
rank whose pooled in-fold AUROC is at or under A, else K. The all-episode curve
(`refit_curve_alldata`) is reported beside it; its erasure was fit on the test folds too, so it
reads biased low and decides nothing. The projector for the rollouts is grown on all episodes
to rank k. The sham is rebuilt at rank k and the four erased and sham rollouts run once at
rank k. `--swap`
adds two rollouts, a class-mean counterfactual (authentic episodes with h <- h - P_k (h - m_surr),
surrogate episodes with m_auth). With the defaults the output is unchanged.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import sys
import time

import numpy as np

from _readout_common import AUTH_BASE, DRIFT, SURR_BASE, fingerprint_extra, load, run_tasks, setup_l3, world_params, write_output

SPEC = "docs/specs/2026-10-15-online-erasure-design.md"
WORLD_BASE = 1_400_000      # the nudge readout's rollout base
GAP_MIN = 0.25
WIN = 24                    # the registered feature window
ERASURE_REF = "artifacts/mechanism/erasure/corrected_l3_h8_wm.json"
INTEGRITY_KEYS = ("frozen_probe_unerased", "frozen_probe_erased", "refit_probe_erased")
INTEGRITY_TOL = 0.01
EXTEND_REL_TOL = 1e-6       # a refit direction with less than this share outside the subspace adds nothing


def _fullrank(t: dict) -> bool:
    return t.get("max_rank", 2) != 2 or t.get("refit_stop") is not None or bool(t.get("swap"))


def run_cell(t: dict, _given: tuple | None = None) -> dict:
    """One (seed, arm) cell. `_given` = (t0, Ha, Hs, rollouts, head) scores rollouts the
    full-rank mode already ran (a, s unerased; ea, es erased; sa, ss sham) on the same pools."""
    if _given is None and _fullrank(t):
        return run_cell_fullrank(t)
    from itasorl import mechanism_readouts as mr
    from itasorl.eval_protocols import readout_from_states
    from itasorl.experiment_b import episode_features
    from itasorl.experiment_b2 import collect_pool, transfer_probe
    from itasorl.novel_readouts import erase, probe_block_directions, projector, sham_projector

    s, g = t["seed"], t["arm"]
    if _given is not None:
        t0, Ha, Hs, v, head = _given
    else:
        setup_l3()
        P = world_params()
        t0 = time.time()
        ag, nm = load(t["agents_dir"], DRIFT, s, g)
        Ha, _ = collect_pool(ag, nm, P, 0.0, t["pool_n"], WIN, "cpu", AUTH_BASE, t["ray_steps"])
        Hs, _ = collect_pool(ag, nm, P, DRIFT, t["pool_n"], WIN, "cpu", SURR_BASE, t["ray_steps"])
        hid = Ha.shape[-1]
        Pr = projector(probe_block_directions(Ha, Hs), rank=2)
        Pv = sham_projector(Pr, 2, s)
        m = 0.5 * (Ha.reshape(-1, hid).mean(0) + Hs.reshape(-1, hid).mean(0))
        # sanity: erasing the pools themselves pins every linear score in the subspace
        resid_pool = float(np.abs((erase(Ha, Pr, m) - m) @ Pr).max())

        def roll(drift, ers=None):
            return mr.rollout_behavior(ag, nm, P, drift, n_eps=t["n_eps"], steps=t["steps"],
                                       seed_base=WORLD_BASE, ray_steps=t["ray_steps"], erase=ers)

        v = {"a": roll(0.0), "s": roll(DRIFT), "ea": roll(0.0, (Pr, m)), "es": roll(DRIFT, (Pr, m)),
             "sa": roll(0.0, (Pv, m)), "ss": roll(DRIFT, (Pv, m))}
        head = {"pool_resid_max": resid_pool}
    common = set(v["a"]["kept"].tolist())
    for x in v.values():
        common &= set(x["kept"].tolist())
    common = sorted(common)
    # Every attempted episode, before the common-survivor intersection (review N04): termination
    # count, mean return summed through termination, mean steps. Post hoc: defined after the
    # first full run had been read, not in an amendment of the erasure spec (its post-run note).
    attempted = {"n_attempted": int(t["n_eps"]),
                 "terminated": {k: int(x["terminated"].sum()) for k, x in v.items()},
                 "pop_return": {k: float(x["ret_all"].mean()) for k, x in v.items()},
                 "pop_steps_alive": {k: float(x["steps_alive"].mean()) for k, x in v.items()}}

    def sub(x, key):
        idx = [list(x["kept"]).index(i) for i in common]
        return x[key][idx]

    B = {k: sub(x, "B") for k, x in v.items()}
    R = {k: sub(x, "ret") for k, x in v.items()}
    Hx = {k: sub(x, "H") for k, x in v.items()}
    out = {"seed": s, "arm": g, "n_common": len(common), **head, **attempted}
    if len(common) < 5:
        out.update(informative=False, reason="fewer than 5 common survivors",
                   seconds=round(time.time() - t0, 1))
        return out
    scale = mr.behavior_scale(B["a"], B["s"])
    real_s, gap = mr.gap_closed(B["s"], B["a"], B["es"], scale)
    sham_s, _ = mr.gap_closed(B["s"], B["a"], B["ss"], scale)
    real_a, _ = mr.gap_closed(B["a"], B["s"], B["ea"], scale)
    sham_a, _ = mr.gap_closed(B["a"], B["s"], B["sa"], scale)
    ii = mr.BEHAVIOR_NAMES.index("intake_rate")
    Xtr = episode_features(np.concatenate([Ha, Hs]))
    ytr = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    yte = np.r_[np.zeros(len(common)), np.ones(len(common))].astype(int)

    from sklearn.metrics import roc_auc_score
    _, _, p_ref = transfer_probe(Xtr, ytr, episode_features(np.concatenate(
        [Hx["a"][:, :WIN], Hx["s"][:, :WIN]])), yte, return_scores=True)
    sd_ref = float(np.std(p_ref))

    def frozen(ka, ks):
        """Frozen-probe AUROC with scores quantized to 1/1000 of the unerased rollouts' score
        spread, so a pinned score (equal up to float32 noise) reads as the tie it is, 0.5,
        rather than a ranking of rounding noise; and the score spread relative to the
        unerased rollouts."""
        _, _, p = transfer_probe(Xtr, ytr, episode_features(np.concatenate(
            [Hx[ka][:, :WIN], Hx[ks][:, :WIN]])), yte, return_scores=True)
        q = np.round(p / (1e-3 * sd_ref)) if sd_ref > 0 else p
        return float(roc_auc_score(yte, q)), float(np.std(p) / sd_ref) if sd_ref > 0 else float("nan")

    out.update({
        "gap": gap, "informative": bool(np.isfinite(gap) and gap >= GAP_MIN),
        "surr_real": real_s, "surr_sham": sham_s, "auth_real": real_a, "auth_sham": sham_a,
        "surr_real_minus_sham": real_s - sham_s, "auth_real_minus_sham": real_a - sham_a,
        "primary": 0.5 * ((real_s - sham_s) + (real_a - sham_a)),
        "return": {k: float(R[k].mean()) for k in R},
        "intake": {k: float(B[k][:, ii].mean()) for k in B},
        "frozen_probe_unerased": frozen("a", "s")[0], "frozen_probe_erased": frozen("ea", "es")[0],
        "frozen_probe_sham": frozen("sa", "ss")[0],
        "score_spread_erased_over_unerased": frozen("ea", "es")[1],
        "score_spread_sham_over_unerased": frozen("sa", "ss")[1],
        # Row i of the erased authentic and surrogate rollouts runs on the same world seed and
        # starts from the same state, so each twin pair is one CV group (experiment_b2.py's
        # pairing rule); split twins bias the refit AUROC toward 0 (correction 2026-10-16).
        "refit_probe_erased": readout_from_states(
            Hx["ea"][:, :WIN], Hx["es"][:, :WIN], seed=s,
            groups=np.r_[np.arange(len(common)), np.arange(len(common))])["target"],
        "seconds": round(time.time() - t0, 1)})
    return out


def basis_of(P: np.ndarray) -> np.ndarray:
    """Orthonormal rows spanning the range of the projector P."""
    w, V = np.linalg.eigh(P)
    return V[:, w > 0.5].T


def extend_basis(B: np.ndarray, D: np.ndarray, room: int, rel_tol: float = EXTEND_REL_TOL) -> tuple[np.ndarray, int]:
    """Add each row of D, projected onto the complement of span(B) (and of the rows already
    added), while room lasts; a row with less than rel_tol of its norm outside is dropped."""
    added = 0
    for d in np.asarray(D, float):
        if added >= room:
            break
        r = d - B.T @ (B @ d)
        nd, nr = float(np.linalg.norm(d)), float(np.linalg.norm(r))
        if nd == 0.0 or nr <= rel_tol * nd:
            continue
        B = np.vstack([B, r / nr])
        added += 1
    return B, added


def refit_auroc(Ha: np.ndarray, Hs: np.ndarray, seed: int, groups: np.ndarray) -> float:
    from itasorl.eval_protocols import readout_from_states
    return float(readout_from_states(Ha, Hs, seed=seed, groups=groups)["target"])


def grow_erasure(Ha: np.ndarray, Hs: np.ndarray, m: np.ndarray, B: np.ndarray, seed: int, max_rank: int,
                 stop: float | None, groups: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Grow the erased subspace from the orthonormal rows B: refit on the erased pools, add the
    refit directions orthogonalized against the subspace, until the twin-grouped refit AUROC
    is at or under `stop` or the rank reaches `max_rank`. Returns the final rows, the refit
    curve by rank and the stop reason."""
    from itasorl.novel_readouts import erase, probe_block_directions
    curve = []
    while True:
        P = B.T @ B
        Ea, Es = erase(Ha, P, m), erase(Hs, P, m)
        auc = refit_auroc(Ea, Es, seed, groups)
        curve.append({"rank": int(len(B)), "refit_auroc": auc})
        if stop is not None and np.isfinite(auc) and auc <= stop:
            return B, curve, "refit at or under the stop value"
        if len(B) >= max_rank:
            return B, curve, "rank cap reached"
        B, added = extend_basis(B, probe_block_directions(Ea, Es), max_rank - len(B))
        if added == 0:
            return B, curve, "no refit direction outside the erased subspace"


def infold_curve(Ha: np.ndarray, Hs: np.ndarray, groups: np.ndarray, max_rank: int) -> list:
    """The stop statistic without test-fold leakage. Per outer fold of the explicit partition
    over `groups`: the rank-two block and every extension are fit on the fold's training rows
    only (midpoint from those rows too), and at each rank the registered probe fit on the erased
    training rows scores the erased test rows. A fold whose growth stops early keeps its last
    subspace for the higher ranks. Returns [{"rank", "refit_auroc", "n_scored"}], the pooled
    out-of-fold AUROC per rank."""
    from sklearn.metrics import roc_auc_score

    from itasorl import folds
    from itasorl.experiment_b import episode_features
    from itasorl.experiment_b2 import transfer_probe
    from itasorl.novel_readouts import erase, probe_block_directions, projector

    H = np.concatenate([Ha, Hs])
    y = np.r_[np.zeros(len(Ha)), np.ones(len(Hs))].astype(int)
    hid = H.shape[-1]
    per_fold = []
    for tr, te in folds.split(groups, 5):
        if len(np.unique(y[te])) < 2:
            continue
        Atr, Str = H[tr][y[tr] == 0], H[tr][y[tr] == 1]
        m = 0.5 * (Atr.reshape(-1, hid).mean(0) + Str.reshape(-1, hid).mean(0))
        B = basis_of(projector(probe_block_directions(Atr, Str), rank=2))
        scores = {}
        while True:
            P = B.T @ B
            _, yv, pv = transfer_probe(episode_features(erase(H[tr], P, m)), y[tr],
                                       episode_features(erase(H[te], P, m)), y[te], return_scores=True)
            scores[int(len(B))] = (np.asarray(yv), np.asarray(pv))
            if len(B) >= max_rank:
                break
            B, added = extend_basis(B, probe_block_directions(erase(Atr, P, m), erase(Str, P, m)),
                                    max_rank - len(B))
            if added == 0:
                break
        per_fold.append(scores)
    curve = []
    for r in sorted({r for sc in per_fold for r in sc}):
        ys, ps = [], []
        for sc in per_fold:
            have = [x for x in sc if x <= r]
            if have:
                yv, pv = sc[max(have)]
                ys.append(yv)
                ps.append(pv)
        yy, pp = np.concatenate(ys), np.concatenate(ps)
        auc = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 and len(pp) == len(yy) else float("nan")
        curve.append({"rank": r, "refit_auroc": auc, "n_scored": int(len(yy))})
    return curve


def choose_k(curve: list, stop: float | None) -> tuple[int, str]:
    """The smallest rank of the in-fold curve at or under `stop`, else its largest rank."""
    if stop is not None:
        for c in curve:
            if np.isfinite(c["refit_auroc"]) and c["refit_auroc"] <= stop:
                return c["rank"], "in-fold refit at or under the stop value"
    return curve[-1]["rank"], ("no stop value: rank cap" if stop is None
                               else "rank cap reached (in-fold refit never at or under the stop value)")


def run_cell_fullrank(t: dict) -> dict:
    """The rank-two cell (unchanged, the integrity gate), then the grown erasure at rank k."""
    from itasorl import mechanism_readouts as mr
    from itasorl.experiment_b2 import collect_pool
    from itasorl.novel_readouts import erase, probe_block_directions, projector, sham_projector

    setup_l3()
    P = world_params()
    s, g = t["seed"], t["arm"]
    t0 = time.time()
    ag, nm = load(t["agents_dir"], DRIFT, s, g)
    Ha, _, ia = collect_pool(ag, nm, P, 0.0, t["pool_n"], WIN, "cpu", AUTH_BASE, t["ray_steps"], return_index=True)
    Hs, _, i_s = collect_pool(ag, nm, P, DRIFT, t["pool_n"], WIN, "cpu", SURR_BASE, t["ray_steps"], return_index=True)
    hid = Ha.shape[-1]
    Pr = projector(probe_block_directions(Ha, Hs), rank=2)
    Pv = sham_projector(Pr, 2, s)
    m = 0.5 * (Ha.reshape(-1, hid).mean(0) + Hs.reshape(-1, hid).mean(0))
    resid_pool = float(np.abs((erase(Ha, Pr, m) - m) @ Pr).max())

    def roll(drift, ers=None):
        return mr.rollout_behavior(ag, nm, P, drift, n_eps=t["n_eps"], steps=t["steps"],
                                   seed_base=WORLD_BASE, ray_steps=t["ray_steps"], erase=ers)

    v = {"a": roll(0.0), "s": roll(DRIFT), "ea": roll(0.0, (Pr, m)), "es": roll(DRIFT, (Pr, m)),
         "sa": roll(0.0, (Pv, m)), "ss": roll(DRIFT, (Pv, m))}
    out = run_cell(t, _given=(t0, Ha, Hs, v, {"pool_resid_max": resid_pool}))

    # episode i of each stored pool shares one CV group (the pooled probe's pair rule)
    groups = np.r_[ia, i_s]
    curve_in = infold_curve(Ha, Hs, groups, t["max_rank"])
    k, reason = choose_k(curve_in, t.get("refit_stop"))
    # the rollout projector: grown on all episodes (no stop), cut at rank k
    B_all, curve_all, reason_all = grow_erasure(Ha, Hs, m, basis_of(Pr), s, t["max_rank"], None, groups)
    B = B_all[:k]
    if len(B) < k:
        reason += f"; the all-episode growth stopped at rank {len(B)} ({reason_all})"
        k = int(len(B))
    Pk = B.T @ B
    Pvk = sham_projector(Pk, k, s)
    vk = {"a": v["a"], "s": v["s"], "ea": roll(0.0, (Pk, m)), "es": roll(DRIFT, (Pk, m)),
          "sa": roll(0.0, (Pvk, m)), "ss": roll(DRIFT, (Pvk, m))}
    rank_k = run_cell(t, _given=(t0, Ha, Hs, vk, {"pool_resid_max": float(np.abs((erase(Ha, Pk, m) - m) @ Pk).max())}))
    rank_k.pop("seconds", None)
    full = {"k": k, "stop_reason": reason, "refit_curve_infold": curve_in,
            "refit_curve_alldata": {"curve": curve_all, "stop_reason": reason_all,
                                    "note": "erasure fit on every episode, the test folds included; "
                                            "biased low, decides nothing"},
            "sham_refit": refit_auroc(erase(Ha, Pvk, m), erase(Hs, Pvk, m), s, groups),
            "rank_k": rank_k,
            "sham_surr_return_drop_population": float(rank_k["pop_return"]["ss"] - rank_k["pop_return"]["s"])}
    if "return" in rank_k:
        full["sham_surr_return_drop"] = float(rank_k["return"]["ss"] - rank_k["return"]["s"])
    if t.get("swap"):
        m_auth, m_surr = Ha.reshape(-1, hid).mean(0), Hs.reshape(-1, hid).mean(0)
        w = {"wa": roll(0.0, (Pk, m_surr)), "ws": roll(DRIFT, (Pk, m_auth))}
        keys = ("a", "s", "sa", "ss")
        rows = [vk[x] for x in keys] + [w["wa"], w["ws"]]
        from run_mechanism_readouts import common_rows
        Ba, Bs, Bsa, Bss, Bwa, Bws = common_rows(*rows)
        swap = {"label": "class-mean counterfactual (not a twin counterfactual)", "n_common": int(len(Ba))}
        if len(Ba) >= 5:
            scale = mr.behavior_scale(Ba, Bs)
            sw_a, gap = mr.gap_closed(Ba, Bs, Bwa, scale)
            sh_a, _ = mr.gap_closed(Ba, Bs, Bsa, scale)
            sw_s, _ = mr.gap_closed(Bs, Ba, Bws, scale)
            sh_s, _ = mr.gap_closed(Bs, Ba, Bss, scale)
            swap.update(gap=gap, informative=bool(np.isfinite(gap) and gap >= GAP_MIN),
                        auth_swap=sw_a, auth_sham=sh_a, surr_swap=sw_s, surr_sham=sh_s,
                        auth_swap_minus_sham=sw_a - sh_a, surr_swap_minus_sham=sw_s - sh_s)
        else:
            swap.update(informative=False, reason="fewer than 5 common survivors")
        full["swap"] = swap
    out["fullrank"] = full
    out["seconds"] = round(time.time() - t0, 1)
    return out


def integrity_failures(rows: list, ref: dict, tol: float = INTEGRITY_TOL) -> list[str]:
    """Gate I: the rank-two frozen and refit probe values per (arm, seed) against the stored
    readout. Both absent or both NaN agree; one finite and one not fails."""
    stored = {(r["arm"], int(r["seed"])): r for r in ref["rows"]}
    fails = []
    for r in rows:
        key = (r["arm"], int(r["seed"]))
        if key not in stored:
            fails.append(f"{key}: not in the stored readout")
            continue
        for q in INTEGRITY_KEYS:
            a, b = float(r.get(q, np.nan)), float(stored[key].get(q, np.nan))
            ok = (np.isnan(a) and np.isnan(b)) or (np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)
            if not ok:
                fails.append(f"{key} {q}: {a!r} vs stored {b!r}")
    return fails


def summarize_fullrank(rows: list, max_rank: int, stop: float | None) -> dict:
    from itasorl.novel_readouts import paired_t90
    out = {}
    for g in ("survival", "untrained"):
        rr = [r for r in rows if r["arm"] == g and "fullrank" in r]
        if not rr:
            continue
        ks = [r["fullrank"]["k"] for r in rr]
        final = [next((c["refit_auroc"] for c in r["fullrank"]["refit_curve_infold"] if c["rank"] == k), np.nan)
                 for r, k in zip(rr, ks)]
        reached = sum(1 for k, a in zip(ks, final)
                      if stop is not None and np.isfinite(a) and a <= stop and k <= max_rank)
        blk = {"k_per_seed": ks, "final_refit_per_seed": final,
               "gate_E": {"seeds_at_or_under_stop": int(reached), "n": len(rr),
                          "passed": bool(reached >= 8), "rule": "in-fold refit at or under the stop in >= 8 "
                                                                f"seeds at rank <= {max_rank}"}}
        if not blk["gate_E"]["passed"]:
            blk["label"] = "VOID (gate E): only k is reported"
            out[g] = blk
            continue
        rk = [r["fullrank"]["rank_k"] for r in rr]
        blk["rank_k"] = summarize(rk).get(g)
        blk["sham_refit"] = paired_t90([r["fullrank"]["sham_refit"] for r in rr])
        inf = [r for r in rr if r["fullrank"]["rank_k"].get("informative")]
        drop = paired_t90([r["fullrank"]["rank_k"]["return"]["ss"] for r in inf],
                          [r["fullrank"]["rank_k"]["return"]["s"] for r in inf])
        base = float(np.mean([r["fullrank"]["rank_k"]["return"]["s"] for r in inf])) if inf else float("nan")
        confounded = bool(np.isfinite(drop["t90"][1]) and drop["t90"][1] < -0.10 * abs(base))
        blk["capacity_check"] = {"sham_surr_return_minus_unerased": drop, "unerased_surr_return_mean": base,
                                 "capacity_confounded": confounded,
                                 "rule": "upper bound below -0.10 x |unerased mean|: no use reading"}
        if any("swap" in r["fullrank"] for r in rr):
            sw = [r["fullrank"]["swap"] for r in rr if r["fullrank"]["swap"].get("informative")]
            d = paired_t90([x["auth_swap_minus_sham"] for x in sw])
            blk["swap"] = {"label": "class-mean counterfactual", "auth_swap_minus_sham": d,
                           "surr_swap_minus_sham": paired_t90([x["surr_swap_minus_sham"] for x in sw]),
                           "swap moves behavior": bool(len(sw) >= 2 and d["mean"] >= 0.10 and d["t90"][0] > 0)}
        if confounded:
            blk["label"] = "no use reading (capacity-confounded)"
        else:
            rd = (blk["rank_k"] or {}).get("readings", {})
            blk["label"] = ("used under removal" if rd.get("1 used under removal")
                            else "costly" if rd.get("2 costly (surrogate return erased minus sham, UB < 0)")
                            else "not used under removal" if rd.get("3 not used under removal")
                            else "none of the frozen readings")
        out[g] = blk
    return out


def summarize(rows: list) -> dict:
    from itasorl.novel_readouts import paired_t90
    out = {}
    for g in ("survival", "untrained"):
        rr = [r for r in rows if r["arm"] == g]
        inf = [r for r in rr if r.get("informative")]
        f = lambda key: paired_t90([r[key] for r in inf])  # noqa: E731
        conds = ("a", "s", "ea", "es", "sa", "ss")
        blk = {"n_informative": len(inf), "scope": "common survivors across all six conditions "
                                                   "(behavior, return and probe readings below)",
               "attrition": {
                   "excluded_fraction": paired_t90([(r["n_attempted"] - r["n_common"]) / r["n_attempted"]
                                                    for r in rr if "n_attempted" in r]),
                   "terminated_per_condition": {k: paired_t90([r["terminated"][k] for r in rr if "terminated" in r])
                                                for k in conds},
                   "note": "attempted episodes per condition and how many ended before the horizon; "
                           "the common-survivor readings drop every episode that ended in any condition"},
               "population": {
                   "surr_return_erased_minus_sham": paired_t90([r["pop_return"]["es"] for r in rr if "pop_return" in r],
                                                               [r["pop_return"]["ss"] for r in rr if "pop_return" in r]),
                   "auth_return_erased_minus_sham": paired_t90([r["pop_return"]["ea"] for r in rr if "pop_return" in r],
                                                               [r["pop_return"]["sa"] for r in rr if "pop_return" in r]),
                   "surr_terminated_erased_minus_sham": paired_t90([r["terminated"]["es"] for r in rr if "terminated" in r],
                                                                   [r["terminated"]["ss"] for r in rr if "terminated" in r]),
                   "auth_terminated_erased_minus_sham": paired_t90([r["terminated"]["ea"] for r in rr if "terminated" in r],
                                                                   [r["terminated"]["sa"] for r in rr if "terminated" in r]),
                   "note": "every attempted episode, return summed through termination; descriptive, "
                           "no pre-stated reading (post hoc, defined after the first full run)"},
               "primary": f("primary"),
               "surr_real_minus_sham": f("surr_real_minus_sham"),
               "auth_real_minus_sham": f("auth_real_minus_sham"),
               "surr_return_erased_minus_sham": paired_t90([r["return"]["es"] for r in inf],
                                                           [r["return"]["ss"] for r in inf]),
               "auth_return_erased_minus_sham": paired_t90([r["return"]["ea"] for r in inf],
                                                           [r["return"]["sa"] for r in inf]),
               "frozen_probe_erased": paired_t90([r.get("frozen_probe_erased", np.nan) for r in rr]),
               "frozen_probe_unerased": paired_t90([r.get("frozen_probe_unerased", np.nan) for r in rr]),
               "refit_probe_erased": paired_t90([r.get("refit_probe_erased", np.nan) for r in rr])}
        held = sum(abs(r.get("frozen_probe_erased", np.nan) - 0.5) < 0.1 for r in rr)
        enough = len(inf) >= 7
        p, sw, aw = blk["primary"], blk["surr_real_minus_sham"], blk["auth_real_minus_sham"]
        blk["readings"] = {
            "enough informative seeds (>= 7)": enough,
            "1 used under removal": bool(enough and p["mean"] >= 0.10 and p["t90"][0] > 0),
            "2 costly (surrogate return erased minus sham, UB < 0)":
                bool(enough and blk["surr_return_erased_minus_sham"]["t90"][1] < 0),
            "3 not used under removal": bool(enough and -0.10 <= sw["t90"][0] and sw["t90"][1] <= 0.10
                                             and -0.10 <= aw["t90"][0] and aw["t90"][1] <= 0.10),
            "4 erasure held (frozen probe within 0.1 of 0.5 in >= 8 seeds)": {"seeds": int(held),
                                                                              "met": bool(held >= 8)},
            "partial erasure (refit probe >= 0.65)": bool(blk["refit_probe_erased"]["mean"] >= 0.65)}
        out[g] = blk
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--arms", nargs="+", default=["survival", "untrained"])
    ap.add_argument("--n-eps", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--pool-n", type=int, default=110)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-rank", type=int, default=2, help="full-rank erasure: rank cap (X10)")
    ap.add_argument("--refit-stop", type=float, default=None,
                    help="full-rank erasure: stop when the refit AUROC is at or under this value (X10)")
    ap.add_argument("--swap", action="store_true", help="full-rank erasure: add the class-mean swap rollouts (X10)")
    ap.add_argument("--erasure-ref", default=ERASURE_REF,
                    help="stored rank-two readout that gates the full-rank mode (within 0.01)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args()
    if a.max_rank < 2:
        ap.error("--max-rank must be at least 2 (the rank-two erasure is the start)")
    ckpt = a.ckpt or a.out + ".ckpt"
    tasks = [{"agents_dir": a.agents_dir, "seed": s, "arm": g, "n_eps": a.n_eps, "steps": a.steps,
              "pool_n": a.pool_n, "ray_steps": a.ray_steps} for g in a.arms for s in a.seeds]
    full = a.max_rank != 2 or a.refit_stop is not None or a.swap
    if full:
        tasks = [{**t, "max_rank": a.max_rank, "refit_stop": a.refit_stop, "swap": bool(a.swap)} for t in tasks]
    if a.dry_run:
        print(f"{len(tasks)} cell(s); full-rank mode {bool(full)} (max rank {a.max_rank}, stop {a.refit_stop}, "
              f"swap {a.swap}); checkpoints in {ckpt}; would write {a.out}")
        return 0
    rows = sorted(run_tasks(run_cell, tasks, ckpt, lambda t: f"{t['arm']}_s{t['seed']}", a.workers,
                            fingerprint_extra(__file__, a.agents_dir, a.seeds, a.arms)),
                  key=lambda r: (r["arm"], r["seed"]))
    body = {"spec": SPEC, "summary": summarize(rows), "rows": rows}
    if full:
        import json
        with open(a.erasure_ref, encoding="utf-8") as fh:
            fails = integrity_failures(rows, json.load(fh))
        if fails:
            print("integrity gate I failed (rank-two values against the stored readout); nothing written:\n  "
                  + "\n  ".join(fails), file=sys.stderr)
            return 1
        body = {"spec": SPEC, "summary": body["summary"],
                "fullrank": {"status": "post hoc, descriptive (X10); changes no stored value or verdict",
                             "settings": {"max_rank": a.max_rank, "refit_stop": a.refit_stop, "swap": bool(a.swap),
                                          "extend_rel_tol": EXTEND_REL_TOL},
                             "integrity": {"reference": a.erasure_ref, "tolerance": INTEGRITY_TOL,
                                           "keys": list(INTEGRITY_KEYS), "gate_I": "PASS"},
                             "summary": summarize_fullrank(rows, a.max_rank, a.refit_stop)},
                "rows": rows}
    write_output(a.out, "scripts/run_erasure_readout.py", sys.argv[1:],
                 {"agents_dir": a.agents_dir, "seeds": a.seeds, "world_base": WORLD_BASE,
                  "n_eps": a.n_eps, "steps": a.steps}, SPEC, body)
    if full:
        import json

        from agent_free_common import provenance
        from promote_bv3_gates import repo_rel
        with open(a.out, encoding="utf-8") as fh:
            doc = json.load(fh)
        p14 = provenance("scripts/run_erasure_readout.py", commit_key="git_commit_at_promotion",
                         output=repo_rel(a.out), device="cpu")
        doc["provenance"].update({k: v for k, v in p14.items() if k != "generated_by"})
        doc["provenance"]["agents_dir"] = repo_rel(a.agents_dir)
        doc["fullrank"]["integrity"]["reference"] = repo_rel(a.erasure_ref)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, default=float)
        for g, blk in doc["fullrank"]["summary"].items():
            print(f"  full rank {g}: k {blk['k_per_seed']}  {blk.get('label')}")
    for k, v in body["summary"]["survival"]["readings"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
