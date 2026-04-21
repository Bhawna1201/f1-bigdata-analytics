"""
Retrain ML Models with Weather-Enhanced Features
==================================================
Retrains all 3 models using the 45-column feature set that includes
10 weather features (track temp, air temp, humidity, rain, etc.)

Compares old vs new performance and saves updated models.

Usage: python ml/retrain_with_weather.py
"""
import logging
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import GOLD_DIR, PROJECT_ROOT

logger = logging.getLogger("retrain_weather")

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Previous results (without weather) for comparison
OLD_RESULTS = {
    "position_predictor": {"mae": 2.74, "rmse": 3.62, "r2": 0.604},
    "podium_classifier": {"accuracy": 0.904, "f1": 0.693, "precision": 0.667, "recall": 0.722},
    "winner_predictor": {"accuracy": 0.500, "top3": 0.896},
}


def load_and_prepare():
    """Load weather-enhanced features and prepare for training."""
    # Use weather-enhanced features
    weather_path = GOLD_DIR / "race_prediction_features_weather.parquet"
    if weather_path.exists():
        df = pd.read_parquet(weather_path)
        logger.info(f"Loaded WEATHER-ENHANCED features: {len(df)} rows, {len(df.columns)} columns")
    else:
        df = pd.read_parquet(GOLD_DIR / "race_prediction_features.parquet")
        logger.warning("Weather features not found — using base features")

    # Encode categoricals
    cat_cols = ["driver_id", "driver_code", "driver_name",
                "constructor_id", "constructor_name",
                "circuit_id", "circuit_name", "race_name"]
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Feature selection
    drop_cols = (cat_cols +
                 ["race_date", "target_position", "target_podium",
                  "target_points_finish", "target_winner", "position"])
    feature_cols = [c for c in df.columns if c not in drop_cols
                    and df[c].dtype in ["float64", "int64", "int32", "float32"]]

    # Identify weather features
    weather_feats = [c for c in feature_cols if c in [
        "avg_track_temp", "max_track_temp", "min_track_temp",
        "avg_air_temp", "max_humidity", "avg_humidity",
        "is_hot_race", "is_cold_race", "temp_variation", "had_rain"
    ]]
    logger.info(f"Total features: {len(feature_cols)} | Weather features: {len(weather_feats)}")
    logger.info(f"Weather features: {weather_feats}")

    # Handle missing values
    X = df[feature_cols].fillna(df[feature_cols].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    # Time-based split: train on <=2024, test on 2025+
    train_mask = df["season"] <= 2024
    test_mask = df["season"] >= 2025

    return df, X, feature_cols, weather_feats, train_mask, test_mask


def train_position_predictor(df, X, feature_cols, train_mask, test_mask):
    """Retrain position predictor with weather features."""
    logger.info("=" * 60)
    logger.info("MODEL 1: Position Predictor (with weather)")
    logger.info("=" * 60)

    y = df["target_position"]
    valid_train = y.notna() & train_mask
    valid_test = y.notna() & test_mask

    logger.info(f"Train: {valid_train.sum()} rows | Test: {valid_test.sum()} rows")

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    model.fit(X[valid_train], y[valid_train])

    y_pred = np.clip(model.predict(X[valid_test]), 1, 20)
    y_true = y[valid_test]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X[y.notna()], y[y.notna()],
                                cv=tscv, scoring="neg_mean_absolute_error")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Compare with old
    old = OLD_RESULTS["position_predictor"]
    logger.info(f"  MAE:  {mae:.2f} (was {old['mae']}, change: {mae - old['mae']:+.2f})")
    logger.info(f"  RMSE: {rmse:.2f} (was {old['rmse']})")
    logger.info(f"  R²:   {r2:.3f} (was {old['r2']})")
    logger.info(f"  CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    logger.info("\n  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"    {row['feature']:35s} {row['importance']:.4f}")

    # Save
    joblib.dump(model, MODELS_DIR / "position_predictor.pkl")
    importance.to_csv(MODELS_DIR / "position_importance.csv", index=False)

    return {
        "mae_old": old["mae"], "mae_new": round(mae, 3),
        "rmse": round(rmse, 3), "r2": round(r2, 3),
        "cv_mae_mean": round(-cv_scores.mean(), 3),
        "improvement": round(old["mae"] - mae, 3),
        "top_features": importance.head(10).to_dict("records"),
    }


def train_podium_classifier(df, X, feature_cols, train_mask, test_mask):
    """Retrain podium classifier with weather features."""
    logger.info("=" * 60)
    logger.info("MODEL 2: Podium Classifier (with weather)")
    logger.info("=" * 60)

    y = df["target_podium"]
    valid_train = y.notna() & train_mask
    valid_test = y.notna() & test_mask

    logger.info(f"Train: {valid_train.sum()} rows | Test: {valid_test.sum()} rows")

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=15, random_state=42
    )
    model.fit(X[valid_train], y[valid_train])

    y_pred = model.predict(X[valid_test])
    y_true = y[valid_test]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    old = OLD_RESULTS["podium_classifier"]
    logger.info(f"  Accuracy:  {acc:.3f} (was {old['accuracy']}, change: {acc - old['accuracy']:+.3f})")
    logger.info(f"  Precision: {prec:.3f} (was {old['precision']})")
    logger.info(f"  Recall:    {rec:.3f} (was {old['recall']})")
    logger.info(f"  F1 Score:  {f1:.3f} (was {old['f1']}, change: {f1 - old['f1']:+.3f})")
    logger.info(f"\n{classification_report(y_true, y_pred)}")

    logger.info("  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"    {row['feature']:35s} {row['importance']:.4f}")

    joblib.dump(model, MODELS_DIR / "podium_classifier.pkl")
    importance.to_csv(MODELS_DIR / "podium_importance.csv", index=False)

    return {
        "accuracy_old": old["accuracy"], "accuracy_new": round(acc, 3),
        "f1_old": old["f1"], "f1_new": round(f1, 3),
        "precision": round(prec, 3), "recall": round(rec, 3),
        "top_features": importance.head(10).to_dict("records"),
    }


def train_winner_predictor(df, X, feature_cols, train_mask, test_mask):
    """Retrain winner predictor with weather features."""
    logger.info("=" * 60)
    logger.info("MODEL 3: Winner Predictor (with weather)")
    logger.info("=" * 60)

    y = df["target_winner"]
    valid_train = y.notna() & train_mask
    valid_test = y.notna() & test_mask

    logger.info(f"Train: {valid_train.sum()} rows | Test: {valid_test.sum()} rows")

    model = GradientBoostingClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model.fit(X[valid_train], y[valid_train])

    # Per-race evaluation
    test_df = df[valid_test].copy()
    test_df["win_prob"] = model.predict_proba(X[valid_test])[:, 1]

    correct = 0
    top3_correct = 0
    total = 0

    for (season, rnd), race in test_df.groupby(["season", "round"]):
        race_sorted = race.sort_values("win_prob", ascending=False)
        actual = race[race["target_winner"] == 1]
        total += 1
        if not actual.empty:
            actual_id = actual.iloc[0].get("driver_id", "")
            if race_sorted.iloc[0].get("driver_id", "") == actual_id:
                correct += 1
            if actual_id in race_sorted.head(3)["driver_id"].values:
                top3_correct += 1

    win_acc = correct / total if total > 0 else 0
    top3_acc = top3_correct / total if total > 0 else 0

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    old = OLD_RESULTS["winner_predictor"]
    logger.info(f"  Races tested: {total}")
    logger.info(f"  Winner accuracy: {win_acc:.1%} (was {old['accuracy']:.1%}, "
                f"change: {(win_acc - old['accuracy'])*100:+.1f}%)")
    logger.info(f"  Top-3 accuracy:  {top3_acc:.1%} (was {old['top3']:.1%}, "
                f"change: {(top3_acc - old['top3'])*100:+.1f}%)")

    # Sample predictions
    logger.info("\n  Sample Predictions (2025-2026):")
    for (season, rnd), race in test_df.groupby(["season", "round"]):
        race_sorted = race.sort_values("win_prob", ascending=False)
        top3 = race_sorted.head(3)
        actual = race[race["target_winner"] == 1]
        winner = actual.iloc[0].get("driver_name", "?") if not actual.empty else "?"
        names = [f"{r.get('driver_name', '?')} ({r['win_prob']:.0%})"
                 for _, r in top3.iterrows()]
        logger.info(f"    {season} R{int(rnd):2d} | Predicted: {', '.join(names)} | Actual: {winner}")
        if rnd >= 5 and season == sorted(test_df["season"].unique())[0]:
            logger.info("    ...")
            break

    logger.info("\n  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"    {row['feature']:35s} {row['importance']:.4f}")

    joblib.dump(model, MODELS_DIR / "winner_predictor.pkl")
    importance.to_csv(MODELS_DIR / "winner_importance.csv", index=False)

    return {
        "accuracy_old": old["accuracy"], "accuracy_new": round(win_acc, 3),
        "top3_old": old["top3"], "top3_new": round(top3_acc, 3),
        "total_races": total, "correct_picks": correct,
        "top_features": importance.head(10).to_dict("records"),
    }


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    logger.info("🏎️  F1 ML Retraining — With Weather Features")
    logger.info("=" * 60)

    df, X, feature_cols, weather_feats, train_mask, test_mask = load_and_prepare()

    results = {}
    results["position_predictor"] = train_position_predictor(df, X, feature_cols, train_mask, test_mask)
    results["podium_classifier"] = train_podium_classifier(df, X, feature_cols, train_mask, test_mask)
    results["winner_predictor"] = train_winner_predictor(df, X, feature_cols, train_mask, test_mask)

    # Save combined results
    results["metadata"] = {
        "retrained_at": datetime.utcnow().isoformat(),
        "total_features": len(feature_cols),
        "weather_features": weather_feats,
        "training_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "seasons_train": "2018-2024",
        "seasons_test": "2025-2026",
    }

    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RETRAINING SUMMARY — WEATHER vs NO WEATHER")
    logger.info("=" * 60)

    p = results["position_predictor"]
    logger.info(f"Position Predictor:  MAE {p['mae_old']} → {p['mae_new']} ({p['improvement']:+.3f})")

    c = results["podium_classifier"]
    logger.info(f"Podium Classifier:   F1  {c['f1_old']} → {c['f1_new']} "
                f"({c['f1_new'] - c['f1_old']:+.3f})")
    logger.info(f"                     Acc {c['accuracy_old']} → {c['accuracy_new']} "
                f"({c['accuracy_new'] - c['accuracy_old']:+.3f})")

    w = results["winner_predictor"]
    logger.info(f"Winner Predictor:    Acc {w['accuracy_old']:.1%} → {w['accuracy_new']:.1%} "
                f"({(w['accuracy_new'] - w['accuracy_old'])*100:+.1f}%)")
    logger.info(f"                     Top3 {w['top3_old']:.1%} → {w['top3_new']:.1%} "
                f"({(w['top3_new'] - w['top3_old'])*100:+.1f}%)")

    logger.info("=" * 60)
    logger.info(f"Results saved: {results_path}")
    logger.info("🏁 Retraining complete")


if __name__ == "__main__":
    main()
