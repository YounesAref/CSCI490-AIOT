# AI Running Twin: Edge-Based Heart Rate Forecasting for Runners

Week 9 midterm prototype for an AI for IoT university project. This Python pipeline predicts a runner's resampled heart rate 30 seconds ahead from the preceding 30 seconds of running measurements. It downloads real FitRec records, prepares features, trains a Random Forest, and compares it with persistence on complete held-out workouts. Edge inference is a later milestone.

## Verified preliminary results

Executed October 22, 2026, with Python 3.9.6 and the pinned packages below. All reportable results use real FitRec data. Artificial data appears only in correctness tests.

| Approach | Test MAE (BPM) | Mean of per-workout MAEs (BPM) |
| --- | ---: | ---: |
| Persistence | 3.3230 | 3.4035 |
| Random Forest | 3.3728 | 3.4254 |

The Random Forest is **0.0498 BPM worse** than persistence on this split. This is a preliminary negative result; no improvement is claimed, and no significance analysis has been performed. The main MAE weights every test window equally. The additional per-workout mean weights workouts equally.

The experiment uses 200 running workouts from 3 users, containing 100,000 original measurement rows. Preprocessing produces 71,312 feature windows. Training uses 160 complete workouts / 56,681 windows; testing uses 40 complete workouts / 14,631 windows. All 3 users occur in both partitions. This evaluates held-out workouts from known users, not unseen runners or future workouts in chronological order.

## Dataset and provenance

- [Official FitRec description](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html), also linked from the [UCSD dataset catalog](https://cseweb.ucsd.edu/~jmcauley/datasets.html#fitrec).
- [Official download directory](https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/).
- File used: `endomondoHR_proper.json` (about 4.9 GB). The downloader stops after selecting the first 200 unique complete records whose `sport` is exactly `run`, in file order. It scanned 378 workouts and saved 4,876,321 bytes locally. This is a convenient, biased subset, not a representative random sample of FitRec.
- Citation: Jianmo Ni, Larry Muhlstein, and Julian McAuley (2019). *Modeling Heart Rate and Activity Data for Personalized Fitness Recommendation*. WWW 2019. [Paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/www19.pdf).
- The authors specify academic use only and request that the dataset not be redistributed or used commercially. Raw data is ignored by Git; obtain it from the authors rather than committing it.

The official `.json` file contains one **Python dictionary literal per line**, rather than strict JSON. The loader uses `ast.literal_eval`, never `eval`, then saves the selected records as genuine JSON Lines. Actual records contain `id`, `userId`, `sport`, `timestamp`, `heart_rate`, `altitude`, `latitude`, `longitude`, `gender`, `url`, and sometimes `speed`. The selected file has unnormalized BPM heart rates; it does not have the normalized `derived_speed`/`tar_heart_rate` schema illustrated for other FitRec representations. Heart rate remains in BPM throughout. Source speed and GPS-derived speed use km/h; altitude uses meters.

The subset has 500 measurements in each workout, irregularly timed. Raw speed is absent in 198 workouts (99,000 / 100,000 measurement slots); GPS is present. Missing speed is therefore derived causally from consecutive GPS positions using great-circle distance divided by elapsed time. No normalized quantities are treated as BPM.

Selection details, source URL, sport counts, and raw-file SHA-256 are recorded in `data/raw/running_workouts.manifest.json`. Feature-file SHA-256, split IDs, seed, model parameters, and library version are stored in `results/split_manifest.json` and the model bundle. Evaluation refuses to use a feature CSV that differs from the one used to train the model.

## Installation

Use Python 3.9–3.12; Python 3.9.6 was used for the recorded run. From the containing workspace:

```bash
cd /Users/younesaref/Desktop/CSCI490AIOT_Project/ai-running-twin
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The original execution used the already installed packages, whose versions exactly match `requirements.txt`. A fresh virtual environment requires network access to install dependencies.

## Reproduce the experiment

With the environment activated, run these commands in the project directory:

```bash
python src/download_data.py --workouts 200
python src/preprocess.py
python src/train.py --seed 42
python src/evaluate.py
python -m pytest -q
```

The raw subset, processed CSV, trained model, metrics, and figures already exist locally. To rerun without downloading, start with `preprocess.py`. Each stage overwrites its own outputs. Rerun training after changing preprocessing or the subset. The evaluation script prints the final MAEs to the terminal. Recorded verification: **12 tests passed**.

If a restricted execution environment blocks the download, allow network access for `download_data.py`. Alternatively, download the official export manually (requires about 4.9 GB of disk space), then select the same subset offline:

```bash
curl --fail --location --output data/raw/endomondoHR_proper.json \
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json
python src/download_data.py --input data/raw/endomondoHR_proper.json --workouts 200
```

The normal streaming command avoids downloading the entire file. `--max-records` defaults to 20,000 scanned records. The source must remain unchanged to reproduce the raw checksum and exact split results.

## Preprocessing methodology

1. Process each workout separately. Reject missing/duplicate workout IDs. Align arrays by index and pad absent/short arrays with missing values. Coerce nonnumeric and infinite measurements to missing. Preserve the original missingness statistics before cleaning or deriving speed.
2. Keep timestamps from 2000 through 2100, heart rate from 30 through 240 BPM, speed from 0 through 60 km/h, altitude from −500 through 9,000 m, and valid geographic coordinates. These are broad engineering plausibility rules. Drop rows lacking valid timestamps or heart rate, sort by time, and average duplicate timestamps.
3. Use recorded valid speed where available. Otherwise calculate haversine distance between the current and preceding GPS positions, divide by their timestamp difference, and convert to km/h. Do not derive across gaps over 30 seconds; reject speeds over 60 km/h. The first GPS point cannot supply a derived speed. No future GPS position is used.
4. Split a workout into separate segments whenever consecutive retained heart-rate measurements are more than 30 seconds apart. Anchor the 10-second grid at the first valid timestamp of the workout. Calculate means in right-closed bins `(t−10, t]`; the bin endpoint is `t`. This is causal aggregation, with no interpolation using future values.
5. Forward-fill at most one empty bin in heart rate, speed, or elevation, within the same segment. There is no backfill. A carried value may therefore reflect an earlier measurement; missingness is not hidden with future information. Preserve whether each heart-rate bin was actually observed. A target bin must have an observation and is never supplied by forward-fill. Residual missing inputs or heart-rate gaps anywhere from the observation through the target cause rejection.
6. Use three consecutive full 10-second bins for each input, covering `(t−30, t]`, with endpoints `t−20, t−10, t`. Exclude the partial first bin of each segment. Advance windows by 10 seconds. The label is resampled heart rate at the grid endpoint **`t+30`**, with no nearest-time label substitution. Because the signal is resampled, this label is the observed mean in `(t+20, t+30]`, not an instantaneous raw sensor reading at exactly that second.
7. Save one row per usable window with workout/user/segment identifiers and explicit observation-start, observation-end, and target timestamps. Identifiers and timestamps are not model predictors.

| Feature | Definition |
| --- | --- |
| `hr_latest` | Heart-rate mean in the latest 10-second input bin, BPM |
| `hr_mean` | Mean of the three input-bin heart rates, BPM |
| `hr_trend` | Least-squares slope across the three equally spaced heart rates: `(last − first) / 20`, BPM/second |
| `speed_mean` | Mean of the three input-bin speeds, km/h |
| `elevation_change` | Last minus first input-bin elevation mean, meters |
| `elapsed_seconds` | Seconds from first valid workout timestamp to observation endpoint |

`elevation_change` compares the first and last aggregates within the 30-second observation interval. The stored `observation_start` is the beginning of that interval, rather than the first bin endpoint.

The recorded run collapsed 16 duplicate timestamps, derived 98,231 speed values, and left 753 speeds missing before bin filling/window rejection. No recorded numeric measurements violated the plausibility limits. There were 572 large gaps, 772 segments, 99,332 resampled bins, and 19,904 initially empty heart-rate bins. Of 95,378 candidate windows, 24,066 were rejected and 71,312 retained. All 200 workouts supplied usable windows. Full field statistics and numeric distributions are in `results/dataset_summary.json` and [DATASET_EXPLORATION.md](results/DATASET_EXPLORATION.md).

## Models and evaluation

Persistence predicts `hr_latest`. It has no fitted parameters and is evaluated on exactly the same test rows as the Random Forest.

The Random Forest uses scikit-learn `RandomForestRegressor` with 150 trees, maximum depth 12, minimum leaf size 5, seed 42, and one worker. It learns from the six features above. No hyperparameter search, scaling, or test-set model selection is performed. `joblib` saves the fitted model with its metadata; load only model files you trust.

An 80/20 random split of sorted unique workout IDs with seed 42 assigns complete workouts to training/testing. No workout or its overlapping windows occurs in both partitions. Preprocessing operates independently by workout and does not learn population-level transformations from test data. The random split is not chronological, and users overlap across partitions.

`MAE = mean(abs(actual_hr − predicted_hr))`, in BPM. Overlapping test windows are correlated; the 14,631 windows should not be interpreted as independent experiments. `mae_by_workout.csv` provides workout-level errors, and `metrics.csv` also includes their unweighted mean. The time-series figure selects the lexicographically first held-out workout ID and its first valid segment, without selecting for prediction quality.

## Generated artifacts

```text
ai-running-twin/
├── data/raw/running_workouts.jsonl
├── data/raw/running_workouts.manifest.json
├── data/processed/features.csv
├── src/download_data.py
├── src/preprocess.py
├── src/train.py
├── src/evaluate.py
├── models/random_forest.joblib
├── results/metrics.csv
├── results/predictions.csv
├── results/mae_comparison.png
├── results/actual_vs_predicted.png
├── results/dataset_summary.json
├── results/DATASET_EXPLORATION.md
├── results/preprocessing_by_workout.csv
├── results/mae_by_workout.csv
├── results/split_manifest.json
├── tests/test_pipeline.py
├── requirements.txt
├── README.md
├── MIDTERM_PROGRESS.md
└── .gitignore
```

The two PNG figures compare MAE and show actual versus predicted heart rates. `predictions.csv` contains test-only predictions with timestamps and workout IDs. The model and data are saved locally but ignored by Git.

## Limitations

- Source-order selection covers only 3 users (125, 14, and 61 workouts). It provides a working experiment, not broad evidence of generalization.
- The source export is already filtered and contains 500 irregular measurements per workout. This is not an evaluation of the original full-resolution sensor stream.
- Most speed values are derived from GPS; GPS noise, altitude errors, and fixed cleaning thresholds can affect features. The prototype does not model sensor quality or pauses explicitly.
- The target is a 10-second aggregate at a 30-second forecast horizon. There is no claim of instantaneous-heart-rate accuracy or clinical validation.
- Persistence remains slightly better. There is no tuning, confidence interval, chronological holdout, or unseen-user evaluation in this milestone.
- There is no on-device profiling, edge deployment, Garmin integration, frontend, API, or cloud component. The saved Random Forest is a desktop prototype; edge suitability has not been established.

## Next steps: Weeks 10–13

| Week | Planned work |
| --- | --- |
| 10 | Sample more runners, inspect GPS/sensor errors, add chronological and user-held-out evaluation while preserving the current reference experiment. |
| 11 | Diagnose persistence versus Random Forest by workout and heart-rate changes; run modest training-only validation and feature ablations. |
| 12 | Measure model size, inference latency, and memory on the intended edge platform; investigate a smaller model if needed. Prepare a Garmin import adapter if data access is available. |
| 13 | Integrate only validated components, rerun final evaluation, document limitations, and prepare the final report and demonstration. |
