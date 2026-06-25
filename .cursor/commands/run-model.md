# Run the AI

Load a trained model from the registry and paper-execute:

```bash
python -m epoch_ai run --bars 6000 --live-bars 300 \
  --long-threshold 0.5 --short-threshold 0.5
```

Requires a prior `train` (models under `artifacts/models/`).

Programmatic API:

```python
from epoch_ai.services import RuntimeService

runtime = RuntimeService(config)
print(runtime.status())
pred = runtime.predict_market(market_df)
result = runtime.run_session(live_bars=300)
```
