# Config sweep

Run the example hyperparameter sweep:

```bash
python -m epoch_ai tune --sweep config/sweeps/example.yaml --bars 4000 --max-steps 3
```

Results land under `artifacts/sweeps/<experiment_name>/metrics.json`.

Override config ad hoc:

```bash
python -m epoch_ai backtest --set walk_forward.step_size=100 --max-steps 5
```
