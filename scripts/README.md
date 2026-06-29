# Scripts

Deterministic reproduction runners. Execute from the **repo root**:

```bash
python scripts/run_e2e.py --quick     # pytest + all experiments (recorded)
python scripts/run_expA.py            # Experiment A, L1
python scripts/run_expB2.py           # Experiment B-v2 (GPU if available)
python -m itasorl.experiment_b        # Experiment B smoke test
```

See the root `README.md` for the full list.
