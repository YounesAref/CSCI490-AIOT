"""Evaluate saved predictions only on complete held-out workouts."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ai-running-twin-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ai-running-twin-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from preprocess import FEATURES, ROOT
from train import load_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/features.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/random_forest.joblib")
    args = parser.parse_args()
    # Load only locally generated, trusted joblib artifacts.
    bundle = joblib.load(args.model)
    meta = bundle["metadata"]
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != meta["features_sha256"]:
        raise ValueError("Feature CSV changed since training; retrain before evaluation")
    if set(meta["train_workout_ids"]) & set(meta["test_workout_ids"]):
        raise ValueError("Training and test workout IDs overlap")
    frame = load_features(args.input)
    test = frame[frame.workout_id.isin(meta["test_workout_ids"])].copy()
    test = test.sort_values(["workout_id", "target_time"])
    test["actual_hr"] = test.target_hr
    test["baseline_prediction"] = test.hr_latest
    test["random_forest_prediction"] = bundle["model"].predict(test[FEATURES])
    results = ROOT / "results"
    cols = ["workout_id", "user_id", "segment_id", "observation_end", "target_time", "elapsed_seconds", "actual_hr", "baseline_prediction", "random_forest_prediction"]
    test[cols].to_csv(results / "predictions.csv", index=False)
    metrics = []
    workout_metrics = []
    for name, column in [("Persistence", "baseline_prediction"), ("Random Forest", "random_forest_prediction")]:
        mae = mean_absolute_error(test.actual_hr, test[column])
        for wid, group in test.groupby("workout_id"):
            workout_metrics.append({"model": name, "workout_id": wid, "test_windows": len(group), "mae_bpm": mean_absolute_error(group.actual_hr, group[column])})
        macro = sum(m["mae_bpm"] for m in workout_metrics if m["model"] == name) / test.workout_id.nunique()
        metrics.append({"model": name, "mae_bpm": mae, "macro_workout_mae_bpm": macro,
                        "test_windows": len(test), "test_workouts": test.workout_id.nunique()})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(results / "metrics.csv", index=False)
    pd.DataFrame(workout_metrics).to_csv(results / "mae_by_workout.csv", index=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    bars = ax.bar(metrics.model, metrics.mae_bpm, color=["#748295", "#188b89"], width=0.55)
    ax.bar_label(bars, fmt="%.3f BPM", padding=5)
    ax.set(ylabel="Mean absolute error (BPM)", title="30-second heart-rate forecast · held-out workouts", ylim=(0, metrics.mae_bpm.max() * 1.22))
    fig.savefig(results / "mae_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for col, label, color in [("baseline_prediction", "Persistence", "#748295"), ("random_forest_prediction", "Random Forest", "#188b89")]:
        axes[0].scatter(test.actual_hr, test[col], s=7, alpha=0.15, label=label, color=color, rasterized=True)
    low = min(test.actual_hr.min(), test.baseline_prediction.min(), test.random_forest_prediction.min())
    high = max(test.actual_hr.max(), test.baseline_prediction.max(), test.random_forest_prediction.max())
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="Actual heart rate (BPM)", ylabel="Predicted heart rate (BPM)", title="All held-out windows")
    axes[0].legend()
    # Select by ID and segment before inspecting prediction quality.
    first = test[test.workout_id == sorted(test.workout_id.unique())[0]]
    first = first[first.segment_id == first.segment_id.min()]
    minutes = (first.elapsed_seconds + 30) / 60
    axes[1].plot(minutes, first.actual_hr, color="#282e3a", label="Actual", linewidth=1.5)
    axes[1].plot(minutes, first.baseline_prediction, color="#748295", label="Persistence", alpha=0.8)
    axes[1].plot(minutes, first.random_forest_prediction, color="#188b89", label="Random Forest", alpha=0.9)
    axes[1].set(xlabel="Target time since workout start (minutes)", ylabel="Heart rate (BPM)", title=f"First test workout: {first.workout_id.iloc[0]}")
    axes[1].legend()
    fig.savefig(results / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)
    print("\nHeld-out evaluation (window-weighted MAE):")
    print(metrics.to_string(index=False, float_format=lambda n: f"{n:.4f}"))
    difference = metrics.mae_bpm.iloc[1] - metrics.mae_bpm.iloc[0]
    print(f"Random Forest minus persistence: {difference:+.4f} BPM (negative is better).")


if __name__ == "__main__":
    main()
