# Train the AI

Primary training workflow (progressive walk-forward + model registry). **Real exchange
data required** — run `download` first. See `docs/get-started.md` for GPU profiles.

```bash
python -m epoch_ai download --full-history
python -m epoch_ai train --set model.device=cuda
python -m epoch_ai evaluate-holdout
```

Fast smoke (provenanced cache still required):

```bash
python -m epoch_ai download --bars 8000
python -m epoch_ai train --bars 8000 --max-steps 12 --fresh
```

**Pause / resume:** `Ctrl+C` after a completed step, then run the same `train` command again.
Use `train --fresh` to discard the checkpoint and restart from step 0.

**Progress:** `python -m epoch_ai progress` (or `checkpoint status`) — steps done/remaining
without running training. Add `--watch` for a live-updating display while training runs
in another terminal.

**Legacy cache without provenance:**

```bash
python -m epoch_ai download --full-history --force
```

Programmatic API:

```python
from epoch_ai.config.settings import load_config
from epoch_ai.services import TrainingService

service = TrainingService(load_config("config/config.yaml"))
result = service.train(register=True, resume=True)
print(result.model_version)
```

Registry cleanup: `model.retain_versions: 10` in config (default) keeps the 10 newest
`v_*` dirs during training; champion and checkpoint models are always protected.
