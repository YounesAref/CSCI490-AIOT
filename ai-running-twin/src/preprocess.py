"""Clean FitRec runs and create causal 30-second features and t+30 targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["hr_latest", "hr_mean", "hr_trend", "speed_mean", "elevation_change", "elapsed_seconds"]
COLUMNS = ["workout_id", "user_id", "segment_id", "observation_start", "observation_end", "target_time"] + FEATURES + ["target_hr"]
FIELDS = ["timestamp", "heart_rate", "speed", "altitude", "latitude", "longitude"]


def measurements(record):
    """Align arrays by index; pad absent/short fields with NaN, never truncate silently."""
    n = len(record.get("timestamp", []))
    return pd.DataFrame({field: pd.to_numeric(pd.Series(record.get(field, []), dtype=object).reindex(range(n)), errors="coerce") for field in FIELDS})


def clean_and_resample(record, max_gap=30):
    frame = measurements(record).replace([np.inf, -np.inf], np.nan)
    stats = Counter(raw_measurements=len(frame))
    # Broad sensor plausibility rules, not clinical cutoffs. Speed is km/h.
    rules = {"timestamp": frame.timestamp.between(946684800, 4102444800),
             "heart_rate": frame.heart_rate.between(30, 240),
             "speed": frame.speed.between(0, 60),
             "altitude": frame.altitude.between(-500, 9000),
             "latitude": frame.latitude.between(-90, 90),
             "longitude": frame.longitude.between(-180, 180)}
    for field, valid in rules.items():
        stats[f"invalid_{field}"] = int((frame[field].notna() & ~valid).sum())
        frame.loc[~valid, field] = np.nan
    origin = frame.timestamp.min()
    stats["removed_missing_timestamp_or_hr"] = int(frame[["timestamp", "heart_rate"]].isna().any(axis=1).sum())
    frame = frame.dropna(subset=["timestamp", "heart_rate"]).sort_values("timestamp")
    stats["duplicate_timestamps_collapsed"] = int(frame.timestamp.duplicated().sum())
    # At duplicate times mean valid measurements, keeping one timestamp.
    frame = frame.groupby("timestamp", as_index=False).mean()
    stats["clean_measurements"] = len(frame)
    if frame.empty:
        return [], stats
    # Causal GPS speed: distance from the previous valid HR/GPS sample / elapsed time.
    # The first sample and pairs crossing a large gap cannot have derived speed.
    lat, lon = np.radians(frame.latitude), np.radians(frame.longitude)
    hav = np.sin(lat.diff() / 2) ** 2 + np.cos(lat.shift()) * np.cos(lat) * np.sin(lon.diff() / 2) ** 2
    distance_km = 2 * 6371.0088 * np.arcsin(np.sqrt(hav.clip(0, 1)))
    delta = frame.timestamp.diff()
    derived = distance_km / delta * 3600
    derived = derived.where(delta.between(0.001, max_gap) & derived.between(0, 60))
    fill = frame.speed.isna() & derived.notna()
    stats["speed_values_derived_from_gps"] = int(fill.sum())
    frame["speed"] = frame.speed.fillna(derived)
    stats["speed_missing_after_derivation"] = int(frame.speed.isna().sum())
    groups = frame.timestamp.diff().gt(max_gap).cumsum()
    stats["large_gaps"] = int(groups.max())
    segments = []
    for segment_id, segment in frame.groupby(groups):
        segment = segment.copy()
        # A value at endpoint t summarizes raw observations in (t-10, t].
        # Integer bins avoid datetime timezone/rounding surprises.
        segment["bin"] = np.ceil((segment.timestamp - origin) / 10).astype(int)
        grid = segment.groupby("bin")[["heart_rate", "speed", "altitude"]].mean()
        grid = grid.reindex(range(int(grid.index.min()), int(grid.index.max()) + 1))
        grid["target_observed"] = grid.heart_rate.notna()
        stats["empty_hr_bins"] += int(grid.heart_rate.isna().sum())
        # Only past values fill one empty interval; no interpolation or backfill.
        grid[["heart_rate", "speed", "altitude"]] = grid[["heart_rate", "speed", "altitude"]].ffill(limit=1)
        grid["timestamp"] = origin + grid.index.to_numpy() * 10
        grid["elapsed_seconds"] = grid.timestamp - origin
        segments.append((int(segment_id), grid))
        stats["resampled_bins"] += len(grid)
    stats["segments"] = len(segments)
    return segments, stats


def workout_features(record, max_gap=30):
    segments, stats = clean_and_resample(record, max_gap)
    rows = []
    for segment_id, grid in segments:
        values = grid[["heart_rate", "speed", "altitude"]].to_numpy()
        times = grid.timestamp.to_numpy()
        elapsed = grid.elapsed_seconds.to_numpy()
        target_observed = grid.target_observed.to_numpy()
        # Three right-closed 10-second bins cover exactly (t-30,t].
        # Start at index 3 to exclude the partial first bin of a workout/segment.
        for end in range(3, len(grid) - 3):
            stats["candidate_windows"] += 1
            span = values[end - 2:end + 4]
            observed = span[:3]
            # Also reject an unfilled HR gap anywhere between observation and label.
            if (not np.isfinite(observed).all() or not np.isfinite(span[:, 0]).all()
                    or not target_observed[end + 3]):
                stats["rejected_windows_missing_data"] += 1
                continue
            hr = observed[:, 0]
            row = {"workout_id": str(record["id"]), "user_id": str(record.get("userId", "unknown")),
                   "segment_id": segment_id, "observation_start": times[end] - 30,
                   "observation_end": times[end], "target_time": times[end + 3],
                   "hr_latest": hr[-1], "hr_mean": hr.mean(),
                   "hr_trend": float((hr[-1] - hr[0]) / 20),
                   "speed_mean": observed[:, 1].mean(),
                   "elevation_change": observed[-1, 2] - observed[0, 2],
                   "elapsed_seconds": elapsed[end], "target_hr": values[end + 3, 0]}
            rows.append(row)
    stats["feature_windows"] = len(rows)
    return rows, stats


def write_exploration(summary, results):
    exploration = summary["exploration"]
    lines = ["# FitRec subset exploration", "",
             "These statistics describe the downloaded running subset, not the full dataset.", "",
             f"- Complete running workouts: {exploration['selected_workouts']:,}",
             f"- Users: {exploration['selected_users']:,}",
             f"- Original measurement rows: {exploration['measurement_count']:,}",
             f"- Usable feature windows: {summary['preprocessing']['feature_windows']:,}", "",
             "Actual fields: " + ", ".join(f"`{f}`" for f in exploration["available_fields"]) + ".", "",
             "Each record is a workout with metadata and aligned sensor arrays. The official export uses Python dictionary literals; the saved subset is standard JSONL.", "",
             "| Raw measurement field | Missing slots | Missing percent | Workouts without field |", "| --- | ---: | ---: | ---: |"]
    for field in FIELDS:
        lines.append(f"| {field} | {exploration['missing_measurements'][field]:,} | {exploration['missing_percent'][field]:.2f}% | {exploration['workouts_missing_field'][field]} |")
    lines += ["", "Missing speed is derived from previous GPS positions where possible; these raw statistics are measured before that fallback. Heart rate is unnormalized BPM.", "",
              "| Preprocessing count | Value |", "| --- | ---: |"]
    for name, value in summary["preprocessing"].items():
        lines.append(f"| {name} | {value:,} |")
    lines += ["", "See `dataset_summary.json` for numeric distributions and array-length checks, `preprocessing_by_workout.csv` for workout counts, and `data/raw/running_workouts.manifest.json` for exact source/selection provenance.", ""]
    (results / "DATASET_EXPLORATION.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/running_workouts.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/features.csv")
    parser.add_argument("--max-gap", type=int, default=30)
    args = parser.parse_args()
    if args.max_gap < 10:
        parser.error("max-gap must be at least 10 seconds")
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if any(str(r.get("sport", "")).lower() != "run" for r in records):
        raise ValueError("Input must contain running workouts only")
    ids = [str(r.get("id", "")) for r in records]
    if "" in ids or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate workout IDs")
    raw = pd.concat([measurements(r) for r in records], ignore_index=True)
    exploration = {
        "selected_workouts": len(records), "selected_users": len({str(r.get("userId")) for r in records}),
        "measurement_count": len(raw), "available_fields": sorted(set().union(*(r.keys() for r in records))),
        "sport_counts": dict(Counter(r.get("sport") for r in records)),
        "missing_measurements": {f: int(raw[f].isna().sum()) for f in FIELDS},
        "missing_percent": {f: round(float(raw[f].isna().mean() * 100), 4) for f in FIELDS},
        "workouts_missing_field": {f: sum(f not in r for r in records) for f in FIELDS},
        "array_length_mismatches": {f: sum(len(r.get(f, [])) != len(r.get("timestamp", [])) for r in records) for f in FIELDS},
        "numeric_summary": json.loads(raw.describe().to_json()),
    }
    all_rows, total, per_workout = [], Counter(), []
    for record in records:
        rows, counts = workout_features(record, args.max_gap)
        all_rows.extend(rows)
        total.update(counts)
        per_workout.append({"workout_id": record["id"], **counts})
    features = pd.DataFrame(all_rows, columns=COLUMNS)
    if features.empty or features.workout_id.nunique() < 5:
        raise ValueError("Insufficient usable workouts; inspect data or download a larger subset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    summary = {"raw_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "exploration": exploration, "preprocessing": dict(total),
               "workouts_with_features": int(features.workout_id.nunique()),
               "workouts_without_features": len(records) - features.workout_id.nunique(),
               "max_gap_seconds": args.max_gap, "resample_seconds": 10,
               "observation_seconds": 30, "forecast_seconds": 30}
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_exploration(summary, results)
    pd.DataFrame(per_workout).fillna(0).to_csv(results / "preprocessing_by_workout.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
