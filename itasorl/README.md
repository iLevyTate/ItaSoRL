# ITASORL core library

Import from Python or run modules directly:

```bash
python -m itasorl.experiment_b    # Experiment B smoke test
```

| Module | Role |
|--------|------|
| `world.py` | World protocol and specs; a wrapper sketch of the surrogate ladder (L2 to L4 wrappers superseded) |
| `patch_of_earth.py` | PatchOfEarthV0 world implementation; its `drift_mode` implements the L1, L2 and L3 rungs |
| `agent.py` | Recurrent world model (RSSM-lite) |
| `agent_ac.py` | Survival actor-critic (B-v2) |
| `experiment_a.py` / `experiment_a_l2.py` | Detectability oracles |
| `experiment_b.py` | Incidental-detection harness |
| `experiment_b2.py` | Survival-coupled B-v2 pipeline; holds the production matched-pair harness (`matched_pair_recurrent_rollout`) |
| `results_io.py` | End-to-end run recording |

Reproduction scripts live in `../scripts/`.
