"""Live rep counting — re-exports shared rep_counting module."""

import sys
from pathlib import Path

FEATURES_DIR = Path(__file__).resolve().parent.parent / "features"
sys.path.insert(0, str(FEATURES_DIR))

from rep_counting import (  # noqa: E402, F401
    DEFAULTS,
    FS,
    MIN_ROWS,
    ORDER,
    REP_CONFIG,
    RepCounterState,
    check_movement_detected,
    count_reps_in_set,
    find_peak_times_ms,
    get_rep_config,
    peaks_from_df,
    register_peaks,
)

# Backward-compatible alias used by scripts
count_reps_in_df = count_reps_in_set
