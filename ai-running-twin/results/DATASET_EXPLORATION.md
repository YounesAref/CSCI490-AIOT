# FitRec subset exploration

These statistics describe the downloaded running subset, not the full dataset.

- Complete running workouts: 200
- Users: 3
- Original measurement rows: 100,000
- Usable feature windows: 71,312

Actual fields: `altitude`, `gender`, `heart_rate`, `id`, `latitude`, `longitude`, `speed`, `sport`, `timestamp`, `url`, `userId`.

Each record is a workout with metadata and aligned sensor arrays. The official export uses Python dictionary literals; the saved subset is standard JSONL.

| Raw measurement field | Missing slots | Missing percent | Workouts without field |
| --- | ---: | ---: | ---: |
| timestamp | 0 | 0.00% | 0 |
| heart_rate | 0 | 0.00% | 0 |
| speed | 99,000 | 99.00% | 198 |
| altitude | 0 | 0.00% | 0 |
| latitude | 0 | 0.00% | 0 |
| longitude | 0 | 0.00% | 0 |

Missing speed is derived from previous GPS positions where possible; these raw statistics are measured before that fallback. Heart rate is unnormalized BPM.

| Preprocessing count | Value |
| --- | ---: |
| raw_measurements | 100,000 |
| invalid_timestamp | 0 |
| invalid_heart_rate | 0 |
| invalid_speed | 0 |
| invalid_altitude | 0 |
| invalid_latitude | 0 |
| invalid_longitude | 0 |
| removed_missing_timestamp_or_hr | 0 |
| duplicate_timestamps_collapsed | 16 |
| clean_measurements | 99,984 |
| speed_values_derived_from_gps | 98,231 |
| speed_missing_after_derivation | 753 |
| large_gaps | 572 |
| empty_hr_bins | 19,904 |
| resampled_bins | 99,332 |
| segments | 772 |
| candidate_windows | 95,378 |
| feature_windows | 71,312 |
| rejected_windows_missing_data | 24,066 |

See `dataset_summary.json` for numeric distributions and array-length checks, `preprocessing_by_workout.csv` for workout counts, and `data/raw/running_workouts.manifest.json` for exact source/selection provenance.
