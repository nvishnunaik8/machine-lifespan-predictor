"""Central configuration for the C-MAPSS predictive maintenance project."""

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Piecewise-linear RUL target: early-life RUL is clipped at this value.
RUL_CAP = 125

# Rolling windows (in cycles) used for time-based features.
WINDOWS = (5, 10, 20)

# Sensors/settings whose training std is below this are treated as constant.
MIN_STD = 1e-6

# RUL thresholds (cycles) for the dashboard status. These are experimental
# choices for demonstration, NOT industrial safety thresholds.
CRITICAL_RUL = 30
WARNING_RUL = 60

RANDOM_STATE = 42
