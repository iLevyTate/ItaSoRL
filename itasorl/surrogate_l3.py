"""L3 dynamics-level surrogate: a learned velocity law `G_motion`.

The L3 surrogate world runs the authentic physics and the REAL observation model, but
replaces the analytic velocity update `(1 - drag*dt)*vel + a*dt` with a small learned net
trained on AUTHENTIC motion transitions. Crucially `G_motion` is NOT given the true drag,
so it must approximate the dynamics from `(vel, a)` alone - its systematic error is the L3
"generative fingerprint", and the net's capacity controls that error (validated: capacity
is a clean monotone knob for the L2-style residual oracle; see `docs/PREREGISTRATION_L3.md`
section 4 Stage-2 and the section-12 deviation).

What G is, exactly (revision step 10, `surrogate_diagnostics`): a 4 -> h -> h -> 2 ReLU MLP
trained full batch with Adam for a FIXED 300 epochs on 250 x 40 authentic scripted-policy
transitions, with no held-out set and no early stopping. In a uniform-drag world such as
world P the authentic law is linear in (vel, a) with constant coefficients, so withholding
drag withholds a constant, and an ordinary least-squares fit recovers the law to rounding:
the fingerprint is the approximation error of this particular finite-trained network.

The observations come from the real sensor model applied to G's motion, so no channel is
synthetic. That does not make every surrogate state reachable under the authentic law, and
the interoception channels report velocity and acceleration exactly, so an observer of the
agent's own inputs can check the velocity law directly.
"""

from __future__ import annotations

import numpy as np


def collect_authentic_transitions(*, n_eps: int = 250, steps: int = 40, params=None,
                                  ray_steps: int = 5, seed0: int = 0):
    """Run authentic (drift-free) rollouts under the scripted policy and return the motion
    transitions `(X, Y)` where `X = [vel_x, vel_y, a_x, a_y]` (N, 4) and `Y = vel_next`
    (N, 2). Drag is deliberately excluded from `X` (see module docstring)."""
    from .patch_of_earth import PatchOfEarthV0
    from .world import SeedBundle, WorldParams
    from .experiment_b import scripted_policy
    X, Y = [], []
    for i in range(n_eps):
        w = PatchOfEarthV0(params or WorldParams())
        w.ray_steps = ray_steps
        w._log_motion = []
        w.reset(SeedBundle(world=seed0 + i, weather=seed0 + 7000 + i, ecology=seed0 + 13000 + i))
        rng = np.random.default_rng(seed0 + i)
        for _ in range(steps):
            w.step(scripted_policy(rng))
        for vel, a, _drag, vnext in w._log_motion:
            X.append([vel[0], vel[1], a[0], a[1]])
            Y.append([vnext[0], vnext[1]])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


class GMotion:
    """A frozen learned velocity law, callable as the world's `_g_motion` hook:
    `(vel, a, drag) -> vel_next`. Ignores `drag` by design. Runs on CPU (single-step world
    inference); `capacity` (the net's hidden width) is the calibration difficulty knob."""

    def __init__(self, net, norm):
        import torch
        # Extract the MLP weights to numpy so per-step world inference needs NO torch call -
        # a torch forward per world step is a severe bottleneck in a full surrogate rollout.
        lin = [m for m in net.to("cpu").modules() if isinstance(m, torch.nn.Linear)]
        self._W = [m.weight.detach().numpy().astype(np.float32) for m in lin]
        self._b = [m.bias.detach().numpy().astype(np.float32) for m in lin]
        self._xm, self._xs, self._ym, self._ys = norm  # numpy normalization stats

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        h = (np.array([vel[0], vel[1], a[0], a[1]], np.float32) - self._xm) / self._xs
        for i, (W, b) in enumerate(zip(self._W, self._b)):
            h = h @ W.T + b
            if i < len(self._W) - 1:
                h = np.maximum(h, 0.0)                 # ReLU on hidden layers only (matches nn.ReLU)
        return (h * self._ys + self._ym).astype(float)


class GradedGMotion:
    """A convex blend of the authentic velocity law and a frozen `GMotion`, callable as the
    world's `_g_motion` hook `(vel, a, drag) -> vel_next`:

        g_alpha(vel, a, drag) = (1 - alpha) * ((1 - drag*dt)*vel + a*dt) + alpha * g(vel, a, drag)

    `alpha == 1.0` returns exactly `g` (the full L3 surrogate); `alpha == 0.0` returns
    exactly the authentic analytic law of `patch_of_earth.py` (`_integrate_motion`), so a
    graded world at alpha=0 is authentic-vs-authentic and reads the L0 chance floor. Used
    by the H2 substrate-grounding ablation (docs/specs/2026-07-21-h2-substrate-grounding-
    ablation-design.md): dialing alpha 1 -> 0 neutralizes the substrate seam, and the
    incidentally-encoded world-identity signal should collapse to the floor if it is
    substrate-grounded. The endpoints short-circuit so alpha=1 is bit-identical to `g`
    (the determinism gate against the saved dumps) and alpha=0 to the authentic law."""

    def __init__(self, g, alpha: float, dt: float):
        self.g = g
        self.alpha = float(alpha)
        self.dt = float(dt)

    def _true_law(self, vel, a, drag) -> np.ndarray:
        # Exact arithmetic of patch_of_earth.py:177 `(1 - drag*dt)*vel + a*dt`.
        vel = np.asarray([vel[0], vel[1]], float)
        a = np.asarray([a[0], a[1]], float)
        return (1.0 - drag * self.dt) * vel + a * self.dt

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        if self.alpha >= 1.0:
            return self.g(vel, a, drag)
        true = self._true_law(vel, a, drag)
        if self.alpha <= 0.0:
            return true
        return (1.0 - self.alpha) * true + self.alpha * np.asarray(self.g(vel, a, drag), float)


def train_g_motion(*, hidden: int = 8, n_eps: int = 250, steps: int = 40, epochs: int = 300,
                   lr: float = 1e-3, seed: int = 0, params=None, ray_steps: int = 5,
                   device: str = "cpu") -> GMotion:
    """Train `G_motion` on authentic transitions. `hidden` is the single difficulty knob;
    `device` = "cpu" or "cuda" for training (the returned net is moved to CPU for the
    per-step world inference). Returns a frozen `GMotion` callable."""
    import torch
    import torch.nn as nn
    X, Y = collect_authentic_transitions(n_eps=n_eps, steps=steps, params=params,
                                         ray_steps=ray_steps, seed0=seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                        nn.ReLU(), nn.Linear(hidden, 2)).to(device)
    xt = torch.tensor((X - xm) / xs, device=device)
    yt = torch.tensor((Y - ym) / ys, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        ((net(xt) - yt) ** 2).mean().backward()
        opt.step()
    return GMotion(net, (xm, xs, ym, ys))


def surrogate_diagnostics(g, *, params, n_train_eps: int = 250, train_steps: int = 40,
                          train_seed: int = 0, heldout_eps: int = 60, heldout_steps: int = 40,
                          heldout_seed: int = 77, horizons=(1, 5, 10, 24), ray_steps: int = 5) -> dict:
    """What the surrogate is, measured (revision step 10).

    In a uniform-drag world (k_land == k_water, as in world P) the authentic velocity law
    vel' = (1 - drag*dt) vel + a dt is LINEAR in (vel, a) with constant coefficients, so
    withholding drag from G withholds a constant. The linear-fit control fits an ordinary
    least-squares map (vel, a) -> vel' on G's own training transitions: it recovers the law to
    rounding, so a linear surrogate would carry no fingerprint, and G's fingerprint is the
    approximation error of a finite-trained ReLU MLP (fixed epochs, no early stopping).

    Reports one-step RMS error of G and of the linear fit on the training transitions and on
    HELD-OUT authentic transitions (disjoint seeds), and the open-loop ROLLOUT divergence:
    from held-out authentic states, iterate G and the authentic law under the same recorded
    accelerations and report the RMS velocity gap after each horizon. One-step error and
    rollout divergence are different quantities; the second compounds the first."""
    X, Y = collect_authentic_transitions(n_eps=n_train_eps, steps=train_steps, params=params,
                                         ray_steps=ray_steps, seed0=train_seed)
    Xh, Yh = collect_authentic_transitions(n_eps=heldout_eps, steps=heldout_steps, params=params,
                                           ray_steps=ray_steps, seed0=heldout_seed)
    Z = np.c_[X.astype(float), np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(Z, Y.astype(float), rcond=None)

    def rms(pred, target):
        return float(np.sqrt(((np.asarray(pred, float) - target) ** 2).sum(1).mean()))

    def g_pred(Xs):
        return np.stack([np.asarray(g(x[:2], x[2:], None), float) for x in Xs])

    lin_tr = Z @ coef
    lin_ho = np.c_[Xh.astype(float), np.ones(len(Xh))] @ coef
    out = {"uniform_drag": bool(params.k_land == params.k_water),
           "authentic_law": {"vel_coef": 1.0 - params.k_land * params.dt, "acc_coef": params.dt},
           "linear_fit": {"coef_vel_x": coef[0].tolist(), "coef_acc_x": coef[2].tolist(),
                          "rms_train": rms(lin_tr, Y.astype(float)),
                          "rms_heldout": rms(lin_ho, Yh.astype(float))},
           "g_one_step": {"rms_train": rms(g_pred(X), Y.astype(float)),
                          "rms_heldout": rms(g_pred(Xh), Yh.astype(float))},
           "n_train": int(len(X)), "n_heldout": int(len(Xh))}
    # Open-loop rollout divergence along held-out trajectories (same accelerations).
    traj = Xh.reshape(heldout_eps, heldout_steps, 4).astype(float)
    a_dt = params.dt
    c_v = 1.0 - params.k_land * params.dt
    gaps = {}
    for H in horizons:
        if H > heldout_steps:
            continue
        d2 = []
        for e in range(heldout_eps):
            v_true = traj[e, 0, :2].copy()
            v_g = v_true.copy()
            for t in range(H):
                acc = traj[e, t, 2:]
                v_true = c_v * v_true + acc * a_dt
                v_g = np.asarray(g(v_g, acc, None), float)
            d2.append(float(((v_g - v_true) ** 2).sum()))
        gaps[str(H)] = float(np.sqrt(np.mean(d2)))
    out["rollout_rms_velocity_gap"] = gaps
    return out
