# Week 9 Midterm Progress — AI Running Twin

Verified October 22, 2026. This milestone implements and executes a real-data forecasting pipeline; edge deployment remains future work.

## Implemented and successfully run

- Streaming official FitRec downloader, safe Python-literal parsing, running-only selection, and a provenance manifest with SHA-256.
- Workout-level cleaning, duplicate timestamp handling, causal GPS-derived speed, 10-second resampling, bounded forward-fill, gap segmentation, six-feature extraction, and CSV export.
- Thirty-second observation intervals with target timestamps exactly 30 seconds after observation endpoints. Targets are observed 10-second heart-rate bin means; no missing label is invented through filling.
- Persistence baseline and scikit-learn Random Forest with a reproducible 80/20 split of complete workouts; fitted model saved with joblib.
- Held-out MAE evaluation, test prediction export, MAE comparison chart, actual-versus-predicted scatter/time-series figure, per-workout errors, and dataset/preprocessing summaries.
- **12 passing tests** for timing, bin boundaries, causal features, gaps, missing targets, GPS-derived speed, duplicate/invalid values, safe parsing, and complete disjoint workout splits. Python compilation also passed.

## Dataset and preprocessing statistics

| Quantity | Verified value |
| --- | ---: |
| Source workouts scanned | 378 |
| Selected complete running workouts | 200 |
| Users | 3 |
| Raw measurement rows | 100,000 |
| Local selected JSONL bytes | 4,876,321 |
| Raw missing heart rate / timestamp / altitude / GPS | 0 |
| Raw missing speed | 99,000 rows (99%); 198 workouts |
| Duplicate timestamps collapsed | 16 |
| Clean unique measurements | 99,984 |
| Missing speeds derived from previous GPS positions | 98,231 |
| Speeds still missing before resampling/fill | 753 |
| Gaps longer than 30 seconds | 572 |
| Separate contiguous segments | 772 |
| Resampled 10-second bins | 99,332 |
| Initially empty heart-rate bins | 19,904 |
| Candidate / rejected / usable windows | 95,378 / 24,066 / 71,312 |
| Training workouts / windows | 160 / 56,681 |
| Test workouts / windows | 40 / 14,631 |

The subset is the first 200 unique `sport == 'run'` workouts in the official filtered export, not a random population sample. The 3 users contribute 125, 14, and 61 workouts; all occur in both train and test. No workout is shared across partitions. All selected workouts produced features. No raw numeric measurement failed the configured plausibility thresholds.

## Preliminary held-out results

| Model | Window-weighted MAE (BPM) | Mean per-workout MAE (BPM) |
| --- | ---: | ---: |
| Persistence | 3.3230 | 3.4035 |
| Random Forest | 3.3728 | 3.4254 |

The Random Forest performed **0.0498 BPM worse** than persistence. The pipeline works, but this experiment does not establish an ML improvement. The correlated overlapping windows, small user count, and single random split limit the conclusion. No synthetic data was used for these results and no test-set tuning was performed.

## Problems encountered and resolved

- The full filtered export is about 4.9 GB. Streaming stopped after the selected complete runs, avoiding a full download.
- The environment initially blocked network/DNS access. The download succeeded with approved external network access using the official UCSD mirror. The README also contains manual download instructions.
- The `.json` export is Python-literal data, delivered as network bytes. The downloader now decodes bytes and parses literals safely, then writes standard JSONL. Tests cover both network bytes and text.
- Speed arrays are absent in 99% of selected rows. Discarding those workouts initially left only two usable runs. A causal GPS-speed fallback now permits all 200 runs to contribute, with no invented speed measurements.
- Duplicate timestamps, empty bins, and long gaps required explicit handling to keep features causal and targets correctly aligned. No global interpolation or cross-workout windows are used.
- The sandbox blocked default Python/font cache locations. Verification used a temporary Python cache, and plotting now configures temporary Matplotlib/font cache paths.

## What remains unfinished

Broader multi-user sampling, chronological and unseen-runner evaluation, uncertainty analysis, feature ablations, modest validation-based model comparison, and sensor-quality analysis remain. Garmin import, actual edge inference, latency/memory/power measurements, final integration, and final-report conclusions have not been implemented. No frontend, API, or cloud deployment was included.

Weeks 10–13 will address broader evaluation and data quality, model diagnosis, edge profiling and possible Garmin import, then integration and final reporting. Detailed methods, commands, and artifact locations are in [README.md](README.md). The next experiment should preserve this baseline and evaluate improved methods without selecting them on the existing held-out test set.
