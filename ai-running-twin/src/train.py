"""Train a modest Random Forest using an 80/20 split of complete workouts."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from preprocess import FEATURES, ROOT


def split_workouts(frame, seed=42):
    ids = sorted(frame.workout_id.astype(str).unique())
    if len(ids) < 5:
        raise ValueError("Need at least five usable workouts for an 80/20 holdout")
    train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=seed)
    train = frame[frame.workout_id.astype(str).isin(train_ids)].copy()
    test = frame[frame.workout_id.astype(str).isin(test_ids)].copy()
    assert set(train.workout_id).isdisjoint(test.workout_id)
    return train, test


def load_features(path):
    frame = pd.read_csv(path, dtype={"workout_id": str, "user_id": str})
    if not np.isfinite(frame[FEATURES + ["target_hr"]].to_numpy()).all():
        raise ValueError("Features/targets contain missing or non-finite values")
    if not np.allclose(frame.target_time - frame.observation_end, 30, rtol=0, atol=1e-6):
        raise ValueError("Incorrect forecast alignment")
    if not np.allclose(frame.observation_end - frame.observation_start, 30, rtol=0, atol=1e-6):
        raise ValueError("Incorrect observation alignment")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/features.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/random_forest.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = load_features(args.input)
    train, test = split_workouts(frame, args.seed)
    model = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=args.seed, n_jobs=1)
    model.fit(train[FEATURES], train.target_hr)
    metadata = {
        "seed": args.seed, "features": FEATURES, "scikit_learn_version": sklearn.__version__,
        "features_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "train_workout_ids": sorted(train.workout_id.unique()),
        "test_workout_ids": sorted(test.workout_id.unique()),
        "train_workouts": train.workout_id.nunique(), "test_workouts": test.workout_id.nunique(),
        "train_windows": len(train), "test_windows": len(test),
        "train_users": train.user_id.nunique(), "test_users": test.user_id.nunique(),
        "users_shared_between_splits": len(set(train.user_id) & set(test.user_id)),
        "model_parameters": model.get_params(),
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, args.model)
    (ROOT / "results/split_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Trained on {len(train):,} windows / {train.workout_id.nunique()} workouts; held out {len(test):,} windows / {test.workout_id.nunique()} workouts.")


if __name__ == "__main__":
    main()
