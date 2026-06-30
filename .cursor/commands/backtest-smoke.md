# Backtest smoke

Quick end-to-end progressive backtest. Synthetic fallback OK here (not for `train`):

```bash
python -m epoch_ai backtest --bars 8000 --max-steps 12 \
  --set data.use_synthetic_fallback=true
```

For faster runs during development:

```bash
python -m epoch_ai backtest --bars 4000 --max-steps 3 --out artifacts/backtests/smoke
```

Check `artifacts/backtests/metrics.json` and `learning_curve.json` after completion.
