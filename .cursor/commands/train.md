# Train the AI

Primary training workflow (progressive walk-forward + model registry):

```bash
python -m epoch_ai train --bars 16000 --log-predictions
python -m epoch_ai train --bars 4000 --max-steps 5   # fast smoke
```

Programmatic API for future interfaces:

```python
from epoch_ai.config.settings import load_config
from epoch_ai.services import TrainingService

service = TrainingService(load_config("config/config.yaml"))
result = service.train(n_bars=16000, register=True)
print(result.model_version)
```
