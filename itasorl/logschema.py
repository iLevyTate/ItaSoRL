"""
ITASORL - logging schema (the immutable data contract, spec sec. 12 + design doc).

The "record-then-render" paradigm keeps three layers fully decoupled:

  runs/{run_id}/
    manifest.json                  run metadata (seeds, ladder, config, spec hashes)
    steps.parquet                  one row per (agent, timestep) - analysis/probe/oracle source
    fields/{step_idx:08d}.npz      periodic full-field snapshots (large; cadence-controlled)
    activations/{episode:06d}.parquet   sidecar agent latent/recurrent states for the H4 probe
    leakage/                       optional precomputed leakage-feature tables

The ground-truth world label `is_surrogate` lives ONLY in the log, never in obs.
The unit of inference is the run / world / pair - NOT the timestep (design doc).
pyarrow is imported lazily so this module loads without it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Per-step record schema (steps.parquet)  -  column -> (dtype, description)
# ---------------------------------------------------------------------------

STEP_COLUMNS: dict[str, tuple[str, str]] = {
    # --- identity / grouping (resample at these levels, not at the timestep) ---
    "run_id":       ("string", "unique run id"),
    "world_id":     ("string", "world instance id (terrain + seed identity)"),
    "pair_id":      ("int64",  "matched-pair id (links authentic & surrogate branches)"),
    "branch":       ("string", "'authentic' | 'surrogate'"),
    "ladder_level": ("string", "L0..L4 (Level enum value)"),
    "episode_id":   ("int64",  "episode index within run"),
    "agent_id":     ("int64",  "creature id"),
    "species_id":   ("int32",  "genome archetype id"),
    "step_idx":     ("int64",  "step within episode"),
    "t":            ("float64", "simulation time"),

    # --- GROUND TRUTH label: probe/oracle target ONLY; never enters obs ---
    "is_surrogate": ("bool", "world identity label (analysis-only)"),

    # --- action / outcome ---
    "reward":     ("float32", "homeostatic reward (never detection)"),
    "terminated": ("bool", "death / terminal"),
    "truncated":  ("bool", "horizon cutoff"),
    "action":     ("list<float32>", "action vector (action_spec order)"),

    # --- full observation: obs-byte leakage baseline + reconstruction ---
    "obs": ("list<float32>", "observation vector (obs_spec order)"),

    # --- focal physiological / kinematic state (spec sec. 6): oracle features ---
    "x": ("float64", "position x"),
    "y": ("float64", "position y"),
    "vx": ("float64", "velocity x"),
    "vy": ("float64", "velocity y"),
    "heading":   ("float64", "heading (rad)"),
    "energy":    ("float32", "E"),
    "hydration": ("float32", "Hyd"),
    "body_temp": ("float32", "Tb"),
    "age":       ("float32", "age"),
    "alive":     ("bool", "alive flag"),

    # --- local environment scalars (spec sec. 4/5) ---
    "ambient_temp": ("float32", "T at agent"),
    "light":        ("float32", "L at agent"),
    "wetness":      ("float32", "medium wetness at agent"),
    "slope_x":      ("float32", "terrain grad x"),
    "slope_y":      ("float32", "terrain grad y"),

    # --- chaos-robust L2 invariants (spec sec. 10/11): precomputed for convenience ---
    "kinetic_energy":         ("float64", "0.5*m*|v|^2 (momentum-decay invariant)"),
    "energy_budget_residual": ("float64", "dE - (intake - costs); ~0 in authentic"),
    "scent_total_mass":       ("float64", "sum of field C_k mass (conservation check)"),

    # --- leakage-audit channels: must FAIL to predict is_surrogate (spec sec. 11) ---
    "episode_len_so_far": ("int64", "steps elapsed this episode"),
    "reset_count":        ("int64", "resets so far this run"),
    "wall_clock_ns":      ("int64", "logged so the audit can verify timing doesn't leak"),
}


# Sidecar: per-step agent internals for the H4 probe (high-dim; kept out of steps.parquet)
ACTIVATION_COLUMNS: dict[str, tuple[str, str]] = {
    "episode_id": ("int64", ""),
    "agent_id":   ("int64", ""),
    "step_idx":   ("int64", ""),
    "branch":     ("string", "'authentic' | 'surrogate'"),
    "is_surrogate": ("bool", "label (analysis-only)"),
    "h": ("list<float32>", "RSSM recurrent state h_t (probe input)"),
    "z": ("list<float32>", "RSSM stochastic latent z_t (probe input)"),
}


# ---------------------------------------------------------------------------
# JSON run manifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    run_id: str
    experiment: str            # 'A' | 'B' | 'B2' | 'C'
    ladder_level: str          # Level value, e.g. "L1_discretization"
    ladder_params: dict[str, Any]   # {"delta": 0.015625} | {"drift_sigma": 1e-3} | ...
    obs_spec_hash: str
    action_spec_hash: str
    channel_mask: dict[str, bool]   # e.g. {"vision": True, "smell": False, "intero": True}
    seeds: dict[str, int]           # asdict(SeedBundle)
    world_params: dict[str, Any]    # asdict(WorldParams)
    agent_config: dict[str, Any]
    episode_horizon: int
    n_episodes: int
    prefix_steps: int = 0           # matched-pair shared prefix
    branch_steps: int = 0           # matched-pair branch length
    field_snapshot_every: int = 0   # steps; 0 = off
    smallest_effect_size: float | None = None  # SESOI for the L0 equivalence test
    git_commit: str = ""
    world_spec_version: str = "v0"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# File layout helpers
# ---------------------------------------------------------------------------

def run_dir(root: str, run_id: str) -> str:
    return os.path.join(root, run_id)


def ensure_layout(root: str, run_id: str) -> dict[str, str]:
    base = run_dir(root, run_id)
    paths = {
        "root": base,
        "manifest": os.path.join(base, "manifest.json"),
        "steps": os.path.join(base, "steps.parquet"),
        "fields": os.path.join(base, "fields"),
        "activations": os.path.join(base, "activations"),
        "leakage": os.path.join(base, "leakage"),
    }
    for key in ("root", "fields", "activations", "leakage"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def write_manifest(root: str, manifest: RunManifest) -> str:
    paths = ensure_layout(root, manifest.run_id)
    with open(paths["manifest"], "w") as f:
        f.write(manifest.to_json())
    return paths["manifest"]


# ---------------------------------------------------------------------------
# Buffered Parquet writer (lazy pyarrow import)
# ---------------------------------------------------------------------------

_ARROW_TYPES = {
    "string": "string",
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
    "float64": "float64",
    "bool": "bool_",
    "list<float32>": "list_float32",
}


def arrow_schema(columns: dict[str, tuple[str, str]] = STEP_COLUMNS):
    """Build a pyarrow.Schema from a column spec. Imports pyarrow lazily."""
    import pyarrow as pa

    def to_type(dt: str):
        if dt == "list<float32>":
            return pa.list_(pa.float32())
        return getattr(pa, {"bool": "bool_"}.get(dt, dt))()

    return pa.schema([(name, to_type(dt)) for name, (dt, _) in columns.items()])


class StepWriter:
    """Append per-step records, flush in batches to a single Parquet file."""

    def __init__(self, path: str, columns: dict[str, tuple[str, str]] = STEP_COLUMNS, batch_size: int = 4096) -> None:
        import pyarrow.parquet as pq  # lazy

        self._pq = pq
        self.path = path
        self.columns = columns
        self.batch_size = batch_size
        self.schema = arrow_schema(columns)
        self._buf: list[dict[str, Any]] = []
        self._writer = None  # opened on first flush

    def append(self, record: dict[str, Any]) -> None:
        missing = set(self.columns) - set(record)
        if missing:
            raise KeyError(f"record missing columns: {sorted(missing)}")
        self._buf.append({k: record[k] for k in self.columns})
        if len(self._buf) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        import pyarrow as pa

        cols = {name: [row[name] for row in self._buf] for name in self.columns}
        batch = pa.record_batch([pa.array(cols[name]) for name in self.columns], schema=self.schema)
        if self._writer is None:
            self._writer = self._pq.ParquetWriter(self.path, self.schema)
        self._writer.write_batch(batch)
        self._buf.clear()

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "StepWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def new_step_record() -> dict[str, Any]:
    """A zero/empty template with every required column - fill what you log."""
    blank: dict[str, Any] = {}
    for name, (dt, _) in STEP_COLUMNS.items():
        if dt == "string":
            blank[name] = ""
        elif dt == "bool":
            blank[name] = False
        elif dt.startswith("float"):
            blank[name] = 0.0
        elif dt.startswith("int"):
            blank[name] = 0
        elif dt == "list<float32>":
            blank[name] = []
    return blank
