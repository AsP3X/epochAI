"""epoch_ai: a self-contained, self-improving crypto AI trading prediction system.

The package is organised into focused, config-driven modules:

* :mod:`epoch_ai.config`      - Pydantic + YAML configuration.
* :mod:`epoch_ai.data`        - historical/live data acquisition and cleaning.
* :mod:`epoch_ai.features`    - modular feature-engineering pipeline.
* :mod:`epoch_ai.models`      - LightGBM model wrapper, versioning and registry.
* :mod:`epoch_ai.logging_system` - SQLite prediction/outcome logging.
* :mod:`epoch_ai.learning`    - progressive (expanding-window) walk-forward engine.
* :mod:`epoch_ai.backtesting` - backtester and trading metrics.
* :mod:`epoch_ai.execution`   - risk management and (paper) execution.
* :mod:`epoch_ai.tracking`    - optional MLflow experiment tracking.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
