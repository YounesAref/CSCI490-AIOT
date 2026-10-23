"""Tiny artificial fixtures test correctness; they never generate report results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from download_data import parse_record
from preprocess import clean_and_resample, workout_features
from train import split_workouts

START = 1500000000


def workout(times=None):
    times = list(range(0, 101, 10)) if times is None else times
    return {"id": 1, "userId": 2, "sport": "run", "timestamp": [START + t for t in times],
            "heart_rate": [100 + t / 10 for t in times], "speed": [10] * len(times), "altitude": [t / 10 for t in times]}


def test_exact_window_and_forecast_alignment():
    rows, _ = workout_features(workout())
    first = rows[0]
    assert len(rows) == 5
    assert first["observation_start"] == START
    assert first["observation_end"] == START + 30
    assert first["target_time"] == START + 60
    assert first["hr_latest"] == 103
    assert first["hr_mean"] == 102
    assert first["hr_trend"] == pytest.approx(0.1)
    assert first["target_hr"] == 106
    assert first["elevation_change"] == 2


def test_future_cannot_change_past_features():
    record = workout()
    before, _ = workout_features(record)
    record["heart_rate"][4:] = [220] * 7
    record["speed"][4:] = [25] * 7
    record["altitude"][4:] = [500] * 7
    after, _ = workout_features(record)
    for key in ["hr_latest", "hr_mean", "hr_trend", "speed_mean", "elevation_change"]:
        assert before[0][key] == after[0][key]
    assert after[0]["target_hr"] == 220


def test_large_gaps_never_cross_windows_or_targets():
    rows, stats = workout_features(workout(list(range(0, 101, 10)) + list(range(200, 301, 10))))
    assert stats["large_gaps"] == 1
    assert len(rows) == 10
    assert all(r["target_time"] <= START + 100 or r["observation_start"] >= START + 200 for r in rows)


def test_missing_label_is_not_forward_filled():
    record = workout()
    record["heart_rate"][6] = None
    rows, _ = workout_features(record)
    assert not any(r["target_time"] == START + 60 for r in rows)
    assert any(r["target_time"] == START + 70 for r in rows)


def test_bins_are_causal_at_irregular_timestamps():
    record = workout([0, 1, 9, 11, 19, 21, 29, 31, 39, 41, 49, 51, 59, 61, 69])
    record["heart_rate"][3] = 200  # At t=11, cannot affect the t=10 bin.
    segments, _ = clean_and_resample(record)
    grid = segments[0][1]
    assert grid.loc[1, "heart_rate"] == pytest.approx((100.1 + 100.9) / 2)
    assert grid.loc[2, "heart_rate"] == pytest.approx((200 + 101.9) / 2)


def test_missing_speed_at_start_never_backfills():
    record = workout()
    record["speed"][:2] = [None, None]
    rows, _ = workout_features(record)
    assert min(r["observation_start"] for r in rows) == START + 10


def test_observation_uses_only_three_bins_in_past_30_seconds():
    record = workout(list(range(0, 101)))
    # t=0 is outside the first observation (0,30]; an old value cannot affect it.
    record["heart_rate"][0] = 220
    rows, _ = workout_features(record)
    assert rows[0]["observation_start"] == START
    assert rows[0]["hr_mean"] == pytest.approx(101.55)
    assert rows[0]["hr_latest"] == pytest.approx(102.55)


def test_workout_split_is_disjoint_complete_and_reproducible():
    frame = pd.DataFrame({"workout_id": np.repeat([str(n) for n in range(10)], 8)})
    train, test = split_workouts(frame)
    assert len(train) == 64 and len(test) == 16
    assert set(train.workout_id).isdisjoint(test.workout_id)
    assert set(train.index) | set(test.index) == set(frame.index)
    assert set(test.index) == set(split_workouts(frame)[1].index)
    assert train.groupby("workout_id").size().eq(8).all()


def test_small_workout_set_rejected():
    with pytest.raises(ValueError):
        split_workouts(pd.DataFrame({"workout_id": ["1", "2"]}))


def test_missing_speed_derived_from_past_gps_only():
    record = workout()
    del record["speed"]
    record["latitude"] = [0] * 11
    record["longitude"] = [i * 0.0001 for i in range(11)]
    segments, stats = clean_and_resample(record)
    grid = segments[0][1]
    assert np.isnan(grid.loc[0, "speed"])
    assert grid.loc[1, "speed"] == pytest.approx(4.003, abs=0.001)
    assert stats["speed_values_derived_from_gps"] == 10
    record["longitude"][5:] = [1] * 6
    changed, _ = clean_and_resample(record)
    assert changed[0][1].loc[1, "speed"] == grid.loc[1, "speed"]


def test_duplicate_timestamps_and_invalid_measurements():
    record = workout([0, 10, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    record["heart_rate"][1:3] = [110, 120]
    record["heart_rate"][-1] = 999
    segments, stats = clean_and_resample(record)
    assert stats["duplicate_timestamps_collapsed"] == 1
    assert stats["invalid_heart_rate"] == 1
    assert segments[0][1].loc[1, "heart_rate"] == 115


def test_parser_accepts_official_literal_without_executing_code():
    assert parse_record("{'sport': 'run', 'id': 1}")["id"] == 1
    assert parse_record(b"{'sport': 'run', 'id': 1}")["id"] == 1
    assert parse_record('{"sport": "run", "id": 2}')["id"] == 2
    with pytest.raises((ValueError, SyntaxError)):
        parse_record("__import__('os').getcwd()")
