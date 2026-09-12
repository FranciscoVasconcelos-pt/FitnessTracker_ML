"""Compare rep counting: batch peaks vs gap-only vs amplitude gates vs live sim."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from live_features import readings_to_resampled_df  # noqa: E402
from predict_live import LivePredictor  # noqa: E402
from rep_counting import (  # noqa: E402
    RepCounterState,
    check_movement_detected,
    find_peak_times_ms,
    get_rep_config,
    peaks_from_df,
    register_peaks,
)


def load_csv(path):
    df = pd.read_csv(path)
    return [
        {
            "t": float(row["epoch_ms"]),
            "acc_x": float(row["acc_x"]),
            "acc_y": float(row["acc_y"]),
            "acc_z": float(row["acc_z"]),
            "gyro_x": float(row.get("gyro_x") or 0),
            "gyro_y": float(row.get("gyro_y") or 0),
            "gyro_z": float(row.get("gyro_z") or 0),
        }
        for _, row in df.iterrows()
    ]


def count_gap_only(readings, exercise):
    state = RepCounterState()
    state.rep_counting_started_ms = readings[0]["t"]
    cfg = get_rep_config(exercise)
    rep_df = readings_to_resampled_df(readings)
    peaks = peaks_from_df(rep_df, exercise)
    for peak_ms, _amp in sorted(peaks):
        if peak_ms < state.rep_counting_started_ms:
            continue
        if peak_ms in state.counted_peak_times:
            continue
        if state.counted_peak_times:
            if peak_ms - max(state.counted_peak_times) < cfg["min_peak_gap_ms"]:
                continue
        state.counted_peak_times.add(peak_ms)
    return len(state.counted_peak_times)


def count_with_gates(readings, exercise):
    state = RepCounterState()
    state.rep_counting_started_ms = readings[0]["t"]
    rep_df = readings_to_resampled_df(readings)
    if not check_movement_detected(rep_df, exercise):
        return 0
    state.movement_detected = True
    peaks = peaks_from_df(rep_df, exercise)
    return register_peaks(peaks, state, exercise)


def count_simulated_live(readings, exercise, chunk=8, debug=False):
    predictor = LivePredictor()
    predictor.reset_reps(exercise)
    predictor.set_rep_target(10)
    reps = 0
    for i in range(0, len(readings), chunk):
        predictor.add_readings(readings[i : i + chunk])
        result = predictor.predict()
        reps = predictor.rep_count
        if debug and i >= len(readings) - chunk:
            rep_df = readings_to_resampled_df(predictor._rep_readings_for_count())
            print(
                f"  debug: reps={reps} movement={predictor.rep_state.movement_detected} "
                f"resampled={len(rep_df)} status={result.get('status')} "
                f"started_ms={predictor.rep_state.rep_counting_started_ms}"
            )
    return reps


def diagnose_peaks_rejected(readings, exercise):
    """Show how many peaks fail amplitude gates."""
    state = RepCounterState()
    state.rep_counting_started_ms = readings[0]["t"]
    state.movement_detected = True
    cfg = get_rep_config(exercise)
    rep_df = readings_to_resampled_df(readings)
    peaks = peaks_from_df(rep_df, exercise)
    rejected_delta = rejected_ratio = rejected_gap = 0
    for peak_ms, amp in sorted(peaks):
        if amp < cfg["min_peak_delta"]:
            rejected_delta += 1
            continue
        if state.session_max_amplitude > 0:
            if amp < cfg["min_amplitude_ratio"] * state.session_max_amplitude:
                rejected_ratio += 1
                continue
        if state.counted_peak_times:
            if peak_ms - max(state.counted_peak_times) < cfg["min_peak_gap_ms"]:
                rejected_gap += 1
                continue
        state.counted_peak_times.add(peak_ms)
        state.session_max_amplitude = max(state.session_max_amplitude, amp)
    return {
        "total_peaks": len(peaks),
        "accepted": len(state.counted_peak_times),
        "rejected_delta": rejected_delta,
        "rejected_ratio": rejected_ratio,
        "rejected_gap": rejected_gap,
    }


def main():
    phone = ROOT / "data" / "raw" / "phone"
    samples = [
        ("bench", "heavy"),
        ("squat", "heavy"),
        ("dead", "heavy"),
        ("ohp", "heavy"),
        ("row", "heavy"),
        ("rest", "heavy"),
    ]

    print(
        f"{'exercise':8} {'batch':>5} {'gap':>5} {'gates':>5} {'live':>5}  "
        "reject(d/r/g)  file"
    )
    print("-" * 85)

    for exercise, category in samples:
        files = sorted(
            phone.glob(f"user-{exercise}-{category}-*.csv"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not files:
            continue

        path = files[0]
        readings = load_csv(path)
        label = "bench" if exercise == "rest" else exercise
        rep_df = readings_to_resampled_df(readings)
        batch = len(find_peak_times_ms(rep_df, label))
        gap = count_gap_only(readings, label)
        gates = count_with_gates(readings, label)
        live = count_simulated_live(readings, label)
        rej = diagnose_peaks_rejected(readings, label)
        reject_str = (
            f"{rej['rejected_delta']}/{rej['rejected_ratio']}/{rej['rejected_gap']}"
        )
        suffix = "(rest)" if exercise == "rest" else ""
        print(
            f"{exercise:8} {batch:5} {gap:5} {gates:5} {live:5}  "
            f"{reject_str:>13}  {path.name} {suffix}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--debug-dead":
        phone = ROOT / "data" / "raw" / "phone"
        f = sorted(
            phone.glob("user-dead-heavy-*.csv"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )[0]
        count_simulated_live(load_csv(f), "dead", debug=True)
    else:
        main()
