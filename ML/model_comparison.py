"""
F1 ML Model Comparison — 3 Tree-Based Ensembles
==================================================
Compares 3 models across all prediction tasks:
  1. Random Forest (Bagging)
  2. Gradient Boosting (Boosting)
  3. XGBoost (Advanced Boosting)

Usage: python ml/model_comparison.py
"""
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import GOLD_DIR, PROJECT_ROOT

logger = logging.getLogger("model_comparison")

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INFO = {
    "Random Forest": {
        "type": "Bagging Ensemble",
        "how_it_works": (
            "Builds many independent decision trees on random subsets of data. "
            "Each tree votes on the prediction and the majority/average wins. "
            "Trees are built in PARALLEL — they don't learn from each other."
        ),
        "strengths": "Resistant to overfitting, handles missing values, fast to train",
        "weaknesses": "Cannot extrapolate beyond training data range, less accurate than boosting",
        "analogy": (
            "Like asking 200 different F1 analysts to predict independently, "
            "then averaging their answers. Each analyst only sees part of the data."
        ),
    },
    "Gradient Boosting": {
        "type": "Boosting Ensemble",
        "how_it_works": (
            "Builds trees SEQUENTIALLY — each new tree focuses on correcting "
            "the mistakes of the previous trees. Early trees capture the big patterns, "
            "later trees fix the details. Learning rate controls how much each tree contributes."
        ),
        "strengths": "Very accurate, captures complex patterns, good feature importance",
        "weaknesses": "Slower to train, can overfit if not tuned, sensitive to noise",
        "analogy": (
            "Like an F1 team reviewing each race, identifying what went wrong, "
            "and adjusting strategy for the next race. Each iteration improves on the last."
        ),
    },
    "XGBoost": {
        "type": "Advanced Boosting (Extreme Gradient Boosting)",
        "how_it_works": (
            "Same sequential boosting concept as Gradient Boosting, but with key "
            "optimizations: built-in regularization (L1/L2) to prevent overfitting, "
            "column subsampling (like Random Forest), efficient sparse data handling, "
            "and parallel tree construction within each boosting round."
        ),
        "strengths": "State-of-the-art accuracy, built-in regularization, handles missing values natively",
        "weaknesses": "More hyperparameters to tune, slightly harder to interpret",
        "analogy": (
            "Like Gradient Boosting but with a smarter engineer who also considers "
            "'what if I am wrong?' (regularization) and uses shortcuts to work faster."
        ),
    },
}


def load_data():
    for filename in ["race_prediction_features_complete.parquet",
                     "race_prediction_features_advanced.parquet",
                     "race_prediction_features_weather.parquet",
                     "race_prediction_features.parquet"]:
        path = GOLD_DIR / filename
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            break
    else:
        raise FileNotFoundError("No Gold features found")

    cat_cols = ["driver_id", "driver_code", "driver_name",
                "constructor_id", "constructor_name",
                "circuit_id", "circuit_name", "race_name"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    drop_cols = (cat_cols +
                 ["race_date", "target_position", "target_podium",
                  "target_points_finish", "target_winner", "position"])
    feature_cols = [c for c in df.columns if c not in drop_cols
                    and df[c].dtype in ["float64", "int64", "int32", "float32"]]

    X = df[feature_cols].fillna(df[feature_cols].median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    train_mask = df["season"] <= 2024
    test_mask = df["season"] >= 2025

    logger.info(f"Features: {len(feature_cols)} | Train: {train_mask.sum()} | Test: {test_mask.sum()}")
    return df, X, feature_cols, train_mask, test_mask


def compare_position_models(df, X, feature_cols, train_mask, test_mask):
    logger.info("\n" + "=" * 70)
    logger.info("TASK 1: POSITION PREDICTION — Which position will the driver finish?")
    logger.info("=" * 70)

    y = df["target_position"]
    vt, vte = y.notna() & train_mask, y.notna() & test_mask
    X_tr, y_tr, X_te, y_te = X[vt], y[vt], X[vte], y[vte]
    logger.info(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=42, verbosity=0),
    }

    results = {}
    best_mae, best_name, best_model = float("inf"), "", None

    for name, model in models.items():
        logger.info(f"\n  [{name}] — {MODEL_INFO[name]['type']}")
        model.fit(X_tr, y_tr)
        y_pred = np.clip(model.predict(X_te), 1, 20)

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)

        tscv = TimeSeriesSplit(n_splits=5)
        cv = cross_val_score(model, X[y.notna()], y[y.notna()], cv=tscv, scoring="neg_mean_absolute_error")

        importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

        results[name] = {
            "mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3),
            "cv_mae": round(-cv.mean(), 3), "cv_std": round(cv.std(), 3),
            "top_features": importance.head(10).to_dict("records"),
        }
        logger.info(f"  MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f} | CV MAE: {-cv.mean():.3f}")

        if mae < best_mae:
            best_mae, best_name, best_model = mae, name, model

    joblib.dump(best_model, MODELS_DIR / "position_predictor.pkl")
    pd.DataFrame({"feature": feature_cols, "importance": best_model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(MODELS_DIR / "position_importance.csv", index=False)
    results["best_model"] = best_name
    logger.info(f"\n  🏆 BEST: {best_name} (MAE: {best_mae:.3f})")
    return results


def compare_podium_models(df, X, feature_cols, train_mask, test_mask):
    logger.info("\n" + "=" * 70)
    logger.info("TASK 2: PODIUM CLASSIFICATION — Will the driver finish top 3?")
    logger.info("=" * 70)

    y = df["target_podium"]
    vt, vte = y.notna() & train_mask, y.notna() & test_mask
    X_tr, y_tr, X_te, y_te = X[vt], y[vt], X[vte], y[vte]
    logger.info(f"Train: {len(X_tr)} | Test: {len(X_te)} | Podium rate: {y_te.mean():.1%}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
            random_state=42, verbosity=0, eval_metric="logloss"),
    }

    results = {}
    best_f1, best_name, best_model = 0, "", None

    for name, model in models.items():
        logger.info(f"\n  [{name}] — {MODEL_INFO[name]['type']}")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)

        results[name] = {
            "accuracy": round(acc, 3), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3),
            "top_features": pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(10).to_dict("records"),
        }
        logger.info(f"  Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    joblib.dump(best_model, MODELS_DIR / "podium_classifier.pkl")
    pd.DataFrame({"feature": feature_cols, "importance": best_model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(MODELS_DIR / "podium_importance.csv", index=False)
    results["best_model"] = best_name
    logger.info(f"\n  🏆 BEST: {best_name} (F1: {best_f1:.3f})")
    return results


def compare_winner_models(df, X, feature_cols, train_mask, test_mask):
    logger.info("\n" + "=" * 70)
    logger.info("TASK 3: WINNER PREDICTION — Who will win the race?")
    logger.info("=" * 70)

    y = df["target_winner"]
    vt, vte = y.notna() & train_mask, y.notna() & test_mask
    X_tr, y_tr, X_te, y_te = X[vt], y[vt], X[vte], y[vte]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=20, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            random_state=42, verbosity=0, eval_metric="logloss"),
    }

    results = {}
    best_acc, best_name, best_model = 0, "", None

    for name, model in models.items():
        logger.info(f"\n  [{name}] — {MODEL_INFO[name]['type']}")
        model.fit(X_tr, y_tr)

        test_df = df[vte].copy()
        test_df["win_prob"] = model.predict_proba(X_te)[:, 1]

        correct = top3 = total = 0
        for (s, r), race in test_df.groupby(["season", "round"]):
            rs = race.sort_values("win_prob", ascending=False)
            actual = race[race["target_winner"] == 1]
            total += 1
            if not actual.empty:
                aid = actual.iloc[0].get("driver_id", "")
                if rs.iloc[0].get("driver_id", "") == aid:
                    correct += 1
                if aid in rs.head(3)["driver_id"].values:
                    top3 += 1

        win_acc = correct / total if total > 0 else 0
        top3_acc = top3 / total if total > 0 else 0

        results[name] = {
            "winner_accuracy": round(win_acc, 3), "top3_accuracy": round(top3_acc, 3),
            "correct": correct, "total": total,
            "top_features": pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False).head(10).to_dict("records"),
        }
        logger.info(f"  Winner: {correct}/{total} ({win_acc:.1%}) | Top-3: {top3}/{total} ({top3_acc:.1%})")

        if win_acc > best_acc:
            best_acc, best_name, best_model = win_acc, name, model

    # Sample predictions
    test_df = df[vte].copy()
    test_df["win_prob"] = best_model.predict_proba(X_te)[:, 1]
    logger.info(f"\n  Sample Predictions ({best_name}):")
    count = 0
    for (s, r), race in test_df.groupby(["season", "round"]):
        rs = race.sort_values("win_prob", ascending=False)
        actual = race[race["target_winner"] == 1]
        winner = actual.iloc[0].get("driver_name", "?") if not actual.empty else "?"
        top3_names = [f"{row.get('driver_name', '?')} ({row['win_prob']:.0%})" for _, row in rs.head(3).iterrows()]
        mark = "✓" if rs.iloc[0].get("driver_id", "") == (actual.iloc[0].get("driver_id", "") if not actual.empty else "") else "✗"
        logger.info(f"    {mark} {int(s)} R{int(r):2d} | Predicted: {', '.join(top3_names)} | Actual: {winner}")
        count += 1
        if count >= 10:
            logger.info(f"    ... ({total - 10} more)")
            break

    joblib.dump(best_model, MODELS_DIR / "winner_predictor.pkl")
    pd.DataFrame({"feature": feature_cols, "importance": best_model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(MODELS_DIR / "winner_importance.csv", index=False)
    results["best_model"] = best_name
    logger.info(f"\n  🏆 BEST: {best_name} (Winner: {best_acc:.1%})")
    return results


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    logger.info("🏎️  F1 ML Model Comparison — 3 Tree-Based Ensembles")
    logger.info("=" * 70)

    logger.info("\nMODEL EXPLANATIONS:")
    for name, info in MODEL_INFO.items():
        logger.info(f"\n  {name} ({info['type']}):")
        logger.info(f"    How: {info['how_it_works']}")
        logger.info(f"    Strengths: {info['strengths']}")
        logger.info(f"    Weaknesses: {info['weaknesses']}")
        logger.info(f"    Analogy: {info['analogy']}")

    df, X, feature_cols, train_mask, test_mask = load_data()

    all_results = {}
    all_results["position_prediction"] = compare_position_models(df, X, feature_cols, train_mask, test_mask)
    all_results["podium_classification"] = compare_podium_models(df, X, feature_cols, train_mask, test_mask)
    all_results["winner_prediction"] = compare_winner_models(df, X, feature_cols, train_mask, test_mask)

    all_results["model_descriptions"] = MODEL_INFO
    all_results["metadata"] = {"compared_at": datetime.now().isoformat(), "features": len(feature_cols)}

    with open(MODELS_DIR / "model_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Also save as training_results.json for dashboard compatibility
    dashboard_results = {
        "position_predictor": all_results["position_prediction"][all_results["position_prediction"]["best_model"]],
        "podium_classifier": all_results["podium_classification"][all_results["podium_classification"]["best_model"]],
        "winner_predictor": all_results["winner_prediction"][all_results["winner_prediction"]["best_model"]],
    }
    with open(MODELS_DIR / "training_results.json", "w") as f:
        json.dump(dashboard_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 70)

    pos = all_results["position_prediction"]
    logger.info("\n  POSITION PREDICTION (lower MAE = better):")
    for n, r in sorted(pos.items(), key=lambda x: x[1].get("mae", 99) if isinstance(x[1], dict) and "mae" in x[1] else 99):
        if isinstance(r, dict) and "mae" in r:
            m = " 🏆" if n == pos["best_model"] else ""
            logger.info(f"    {n:25s} MAE: {r['mae']:.3f}  RMSE: {r['rmse']:.3f}  R²: {r['r2']:.3f}{m}")

    pod = all_results["podium_classification"]
    logger.info("\n  PODIUM CLASSIFICATION (higher F1 = better):")
    for n, r in sorted(pod.items(), key=lambda x: -x[1].get("f1", 0) if isinstance(x[1], dict) and "f1" in x[1] else 0):
        if isinstance(r, dict) and "f1" in r:
            m = " 🏆" if n == pod["best_model"] else ""
            logger.info(f"    {n:25s} F1: {r['f1']:.3f}  Acc: {r['accuracy']:.3f}  Prec: {r['precision']:.3f}  Rec: {r['recall']:.3f}{m}")

    win = all_results["winner_prediction"]
    logger.info("\n  WINNER PREDICTION (higher accuracy = better):")
    for n, r in sorted(win.items(), key=lambda x: -x[1].get("winner_accuracy", 0) if isinstance(x[1], dict) and "winner_accuracy" in x[1] else 0):
        if isinstance(r, dict) and "winner_accuracy" in r:
            m = " 🏆" if n == win["best_model"] else ""
            logger.info(f"    {n:25s} Winner: {r['winner_accuracy']:.1%}  Top-3: {r['top3_accuracy']:.1%}  ({r['correct']}/{r['total']}){m}")

    logger.info(f"\n  WHY THESE 3 MODELS?")
    logger.info(f"    Random Forest  = Bagging (parallel trees, majority vote)")
    logger.info(f"    Gradient Boost = Boosting (sequential trees, learn from mistakes)")
    logger.info(f"    XGBoost        = Advanced Boosting (regularized, optimized)")
    logger.info(f"    All three are tree-based ensembles — the gold standard for tabular data.")

    logger.info("\n🏁 Comparison complete")


if __name__ == "__main__":
    main()
