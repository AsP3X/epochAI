# Paper-trade demo

Exercise the execution path on near-random data (forces directional positions):

```bash
python -m epoch_ai paper-trade --bars 6000 --live-bars 300 \
  --long-threshold 0.5 --short-threshold 0.5
```

With inline retrain every 50 bars:

```bash
python -m epoch_ai paper-trade --bars 6000 --live-bars 300 \
  --retrain-every 50 --long-threshold 0.5 --short-threshold 0.5
```
