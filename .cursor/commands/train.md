# Train the AI

Primary training workflow (progressive walk-forward + model registry):

```bash
python -m epoch_ai train --bars 16000 --log-predictions
python -m epoch_ai train --bars 4000 --max-steps 5   # fast smoke
```

**Pause / resume:** `Ctrl+C` after a completed step, then run the same `train` command again.
Use `train --fresh` to discard the checkpoint and restart from step 0.

**Seed checkpoint** (runs that pre-date auto-checkpointing):

```bash
python -m epoch_ai checkpoint seed --last-step 75
python -m epoch_ai train --log-predictions
```

Programmatic API for future interfaces:

```python
from epoch_ai.config.settings import load_config
from epoch_ai.services import TrainingService

service = TrainingService(load_config("config/config.yaml"))
result = service.train(n_bars=16000, register=True, resume=True)
print(result.model_version)
```

Registry cleanup: `model.retain_versions: 10` in config (default) keeps the 10 newest
`v_*` dirs during training; champion and checkpoint models are always protected.
