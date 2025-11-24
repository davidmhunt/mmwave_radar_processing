# Logging Plan

Plan for centralized logging across `mmwave_radar_processing`, aligned with the new GUI and runtime scripts.

## Scope and constraints
- Stdout-only logging (no file handlers/rotation for now).
- Console output is sufficient; no in-GUI log panel in v1.
- Default log level for the GUI/scripts is INFO (overridable via CLI); library code can emit logs once the logger is configured.
- Logger helper lives in `mmwave_radar_processing/logging/logger.py`; reuse it everywhere.

## Logger helper design (`logging/logger.py`)
- `setup_logger(name="mmwave_radar_processing", level=logging.INFO, set_default=True)`: builds a logger, clears existing handlers, attaches a single `StreamHandler` to stdout with format `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s` and `datefmt="%H:%M:%S"`, disables propagation, caches as default.
- `get_logger(name=None)`: returns the default logger (initializing via `setup_logger` if needed) or a named logger; if the named logger lacks handlers, it is configured with the default formatter/level.
- Classes should accept a `logger` parameter (defaulting to `get_logger(__name__)`) rather than module-level globals. Runtime scripts may use module-level loggers after calling `setup_logger`.

## Configuration flow
- CLI entry points (e.g., `scripts/launch_mmwave_viewer.py`, future scripts) parse `--log-level` (DEBUG/INFO/WARNING/ERROR) and call `setup_logger(level=level)` once at startup.
- GUI launch (via `launch_mmwave_viewer.py`) defaults to INFO if no flag is provided.
- Library code imports `get_logger` and passes a logger into classes on construction; avoid re-calling `setup_logger` inside libraries.

## Usage patterns
```python
from mmwave_radar_processing.logging.logger import setup_logger, get_logger

# In script entry point
setup_logger(level=logging.INFO)
logger = get_logger(__name__)
logger.info("starting app")

# In a class
class RangeAngleProcessor:
    def __init__(self, config_manager, logger=None, num_angle_bins=64):
        self.logger = logger or get_logger(__name__)
        ...
    def process(self, adc_cube, **kwargs):
        self.logger.debug("processing ADC cube")
```

Guidelines:
- One `setup_logger` per process, at startup. Do not configure handlers inside libraries.
- Use `get_logger(__name__)` for namespacing; rely on the shared formatter/level.
- Keep propagation disabled unless explicitly adding parent/root handlers later.

## Integration points
- Controllers: log lifecycle events (dataset load, processor init, play/pause, export start/finish, errors).
- Processors: log configuration parameters at init, and WARN/ERROR on invalid inputs.
- Models (CpslDS/ConfigManager wrappers): log dataset/config load outcomes.
- Scripts: accept `--log-level`; pass loggers into constructed classes.
- Cleanup plan: replace existing `print` statements in `mmwave_radar_processing/point_cloud_processing`, `mmwave_radar_processing/processors`, and `mmwave_radar_processing/supportFns` with logger calls (`logger.info/debug/warning/error`) using injected loggers per class/function. Track this as part of the logging integration pass.

## Future extensions (planned, not in v1)
- Optional file/rotating handlers for long-running sessions.
- Optional in-GUI log pane that subscribes to log records via a custom handler.
