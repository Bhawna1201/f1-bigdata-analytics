"""
Phase 2: F1 Machine Learning Models
====================================
1. XGBoost Race Position Predictor (regression)
2. Podium Classifier (binary classification)
3. Tire Degradation Model (per-compound curve fitting)

Usage:
    python ml/train_models.py
"""
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import joblib

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import GOLD_DIR, PROJECT_ROOT

logger = logging.getLogger("ml_training")

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────

def load_and_prepare_data():
    """Load Gold features and prepare train/test splits."""
    df = pd.read_parquet(GOLD_DIR / "race_prediction_features.parquet")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Encode categorical columns ──
    cat_cols = ["driver_id", "driver_code", "driver_name",
                "constructor_id", "constructor_name",
                "circuit_id", "circuit_name", "race_name"]

    label_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # ── Feature selection ──
    # Drop non-feature columns
    drop_cols = (cat_cols +
                 ["race_date", "target_position", "target_podium",
                  "target_points_finish", "target_winner", "position"])

    feature_cols = [c for c in df.columns if c not in drop_cols
                    and df[c].dtype in ["float64", "int64", "int32", "float32"]]

    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── Handle missing values ──
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # ── Time-based split: train on 2018-2023, test on 2024-2025 ──
    if "season" in df.columns:
        train_mask = df["season"] <= 2023
        test_mask = df["season"] >= 2024
    else:
        train_mask = np.arange(len(df)) < int(len(df) * 0.8)
        test_mask = ~train_mask

    return df, X, feature_cols, label_encoders, train_mask, test_mask


# ─────────────────────────────────────────────────────────
# Model 1: Race Position Predictor (Regression)
# ─────────────────────────────────────────────────────────

def train_position_predictor(df, X, feature_cols, train_mask, test_mask):
    """XGBoost regression to predict finishing position."""
    logger.info("=" * 60)
    logger.info("MODEL 1: Race Position Predictor (Regression)")
    logger.info("=" * 60)

    y = df["target_position"].copy()

    # Drop rows with no target
    valid = y.notna() & train_mask
    valid_test = y.notna() & test_mask

    X_train, y_train = X[valid], y[valid]
    X_test, y_test = X[valid_test], y[valid_test]

    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Train with GradientBoosting (sklearn, no xgboost dependency) ──
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 20)  # Positions are 1-20

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE:  {mae:.2f} positions")
    logger.info(f"RMSE: {rmse:.2f} positions")
    logger.info(f"R²:   {r2:.3f}")

    # ── Cross-validation (time series split) ──
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X[y.notna()], y[y.notna()],
                                cv=tscv, scoring="neg_mean_absolute_error")
    logger.info(f"CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # ── Feature importance ──
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    logger.info("\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:35s} {row['importance']:.4f}")

    # ── Save ──
    joblib.dump(model, MODELS_DIR / "position_predictor.pkl")
    importance.to_csv(MODELS_DIR / "position_importance.csv", index=False)

    results = {
        "model": "GradientBoostingRegressor",
        "target": "finishing_position",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
        "cv_mae_mean": round(-cv_scores.mean(), 3),
        "cv_mae_std": round(cv_scores.std(), 3),
        "top_features": importance.head(10).to_dict("records"),
        "trained_at": datetime.utcnow().isoformat(),
    }

    return model, results


# ─────────────────────────────────────────────────────────
# Model 2: Podium Classifier (Binary Classification)
# ─────────────────────────────────────────────────────────

def train_podium_classifier(df, X, feature_cols, train_mask, test_mask):
    """Predict whether a driver finishes on the podium (top 3)."""
    logger.info("=" * 60)
    logger.info("MODEL 2: Podium Classifier (Binary)")
    logger.info("=" * 60)

    y = df["target_podium"].copy()

    valid = y.notna() & train_mask
    valid_test = y.notna() & test_mask

    X_train, y_train = X[valid], y[valid]
    X_test, y_test = X[valid_test], y[valid_test]

    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    logger.info(f"Podium rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"Accuracy:  {acc:.3f}")
    logger.info(f"Precision: {prec:.3f}")
    logger.info(f"Recall:    {rec:.3f}")
    logger.info(f"F1 Score:  {f1:.3f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # ── Feature importance ──
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    logger.info("\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:35s} {row['importance']:.4f}")

    # ── Save ──
    joblib.dump(model, MODELS_DIR / "podium_classifier.pkl")
    importance.to_csv(MODELS_DIR / "podium_importance.csv", index=False)

    results = {
        "model": "GradientBoostingClassifier",
        "target": "podium_finish",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1_score": round(f1, 3),
        "podium_rate_train": round(float(y_train.mean()), 3),
        "podium_rate_test": round(float(y_test.mean()), 3),
        "top_features": importance.head(10).to_dict("records"),
        "trained_at": datetime.utcnow().isoformat(),
    }

    return model, results


# ─────────────────────────────────────────────────────────
# Model 3: Winner Predictor (per-race probability ranking)
# ─────────────────────────────────────────────────────────

def train_winner_predictor(df, X, feature_cols, train_mask, test_mask):
    """Predict race winner — output probabilities, rank per race."""
    logger.info("=" * 60)
    logger.info("MODEL 3: Race Winner Predictor (Probability Ranking)")
    logger.info("=" * 60)

    y = df["target_winner"].copy()

    valid = y.notna() & train_mask
    valid_test = y.notna() & test_mask

    X_train, y_train = X[valid], y[valid]
    X_test, y_test = X[valid_test], y[valid_test]

    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    logger.info(f"Win rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    model = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Per-race evaluation: pick highest-probability driver ──
    test_df = df[valid_test].copy()
    test_df["win_probability"] = model.predict_proba(X_test)[:, 1]

    correct_picks = 0
    total_races = 0
    top3_correct = 0

    for (season, rnd), race in test_df.groupby(["season", "round"]):
        race_sorted = race.sort_values("win_probability", ascending=False)
        predicted_winner = race_sorted.iloc[0]
        actual_winner = race[race["target_winner"] == 1]

        total_races += 1
        if not actual_winner.empty:
            actual_id = actual_winner.iloc[0].get("driver_id", "")
            pred_id = predicted_winner.get("driver_id", "")
            if actual_id == pred_id:
                correct_picks += 1
            # Check if actual winner was in top 3 predicted
            top3_ids = race_sorted.head(3)["driver_id"].values if "driver_id" in race_sorted.columns else []
            if actual_id in top3_ids:
                top3_correct += 1

    win_accuracy = correct_picks / total_races if total_races > 0 else 0
    top3_accuracy = top3_correct / total_races if total_races > 0 else 0

    logger.info(f"Races evaluated: {total_races}")
    logger.info(f"Correct winner picks: {correct_picks}/{total_races} ({win_accuracy:.1%})")
    logger.info(f"Winner in top-3 predicted: {top3_correct}/{total_races} ({top3_accuracy:.1%})")

    # ── Feature importance ──
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    logger.info("\nTop 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']:35s} {row['importance']:.4f}")

    # ── Show sample predictions for recent races ──
    logger.info("\n── Sample Predictions (2024-2025 races) ──")
    for (season, rnd), race in test_df.groupby(["season", "round"]):
        race_sorted = race.sort_values("win_probability", ascending=False)
        top3 = race_sorted.head(3)
        actual_winner = race[race["target_winner"] == 1]
        winner_name = actual_winner.iloc[0].get("driver_name", "?") if not actual_winner.empty else "?"

        names = []
        for _, r in top3.iterrows():
            name = r.get("driver_name", f"Driver")
            prob = r["win_probability"]
            names.append(f"{name} ({prob:.0%})")

        logger.info(f"  {season} R{int(rnd):2d} | Predicted: {', '.join(names)} | Actual: {winner_name}")

        # Only show first 10 races
        if rnd >= 10 and season == list(test_df["season"].unique())[0]:
            logger.info("  ...")
            break

    # ── Save ──
    joblib.dump(model, MODELS_DIR / "winner_predictor.pkl")
    importance.to_csv(MODELS_DIR / "winner_importance.csv", index=False)

    results = {
        "model": "GradientBoostingClassifier",
        "target": "race_winner",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "correct_winner_picks": correct_picks,
        "total_races": total_races,
        "winner_accuracy": round(win_accuracy, 3),
        "winner_in_top3_accuracy": round(top3_accuracy, 3),
        "top_features": importance.head(10).to_dict("records"),
        "trained_at": datetime.utcnow().isoformat(),
    }

    return model, results


# ─────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    logger.info("🏎️  F1 ML Training Pipeline — Phase 2")
    logger.info("=" * 60)

    # Load data
    df, X, feature_cols, encoders, train_mask, test_mask = load_and_prepare_data()

    all_results = {}

    # Model 1: Position predictor
    _, results1 = train_position_predictor(df, X, feature_cols, train_mask, test_mask)
    all_results["position_predictor"] = results1

    # Model 2: Podium classifier
    _, results2 = train_podium_classifier(df, X, feature_cols, train_mask, test_mask)
    all_results["podium_classifier"] = results2

    # Model 3: Winner predictor
    _, results3 = train_winner_predictor(df, X, feature_cols, train_mask, test_mask)
    all_results["winner_predictor"] = results3

    # ── Save combined results ──
    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved: {results_path}")

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Position Predictor — MAE: {results1['mae']:.2f} positions, R²: {results1['r2']:.3f}")
    logger.info(f"Podium Classifier  — F1: {results2['f1_score']:.3f}, Precision: {results2['precision']:.3f}")
    logger.info(f"Winner Predictor   — Accuracy: {results3['winner_accuracy']:.1%}, "
                f"Top-3: {results3['winner_in_top3_accuracy']:.1%}")
    logger.info("=" * 60)
    logger.info("🏁 Phase 2 complete — models saved in /models/")


if __name__ == "__main__":
    main()
