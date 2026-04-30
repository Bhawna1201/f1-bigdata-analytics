"""
Phase 3: F1 Agentic AI Layer
==============================
Four autonomous agents orchestrated by LangGraph:
  1. Data Agent — monitors APIs, self-heals ingestion
  2. Feature Agent — proposes and tests new features
  3. Model Agent — evaluates and improves ML models
  4. Insight Agent — generates race strategy briefings

Requirements:
    pip install langchain langgraph openai pandas scikit-learn

Usage:
    # Set your API key (OpenAI or Anthropic)
    export OPENAI_API_KEY=sk-...
    # Or use Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

    python agents/agentic_pipeline.py
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, Literal

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from agent_memory import save_agent_memory, load_memory, get_memory_context

logger = logging.getLogger("agentic_pipeline")

# ─── Project paths ───
PROJECT_ROOT = Path(__file__).parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
MODELS_DIR = PROJECT_ROOT / "models"
AGENTS_LOG = PROJECT_ROOT / "logs" / "agents"
AGENTS_LOG.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# Shared Agent State (flows through the entire pipeline)
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # Data Agent outputs
    data_health: dict
    ingestion_status: str
    data_issues: list

    # Feature Agent outputs
    new_features_proposed: list
    new_features_accepted: list
    feature_test_results: dict

    # Model Agent outputs
    model_performance: dict
    model_recommendations: list
    retrain_triggered: bool

    # Insight Agent outputs
    race_briefing: str
    predictions: dict

    # Orchestrator
    phase: str
    errors: list
    timestamp: str


# ═══════════════════════════════════════════════════════════
# Agent 1: Data Agent — Monitor, Validate, Self-Heal
# ═══════════════════════════════════════════════════════════

def data_agent(state: AgentState) -> AgentState:
    """
    Monitors data quality and API health.
    - Checks for missing data, schema changes, null spikes
    - Validates row counts match expectations
    - Flags anomalies for human review or auto-fixes
    """
    logger.info("🔍 DATA AGENT: Running data health checks...")
    issues = []
    health = {}

    try:
        # Check Gold features exist and are fresh
        gold_path = GOLD_DIR / "race_prediction_features.parquet"
        
        if not gold_path.exists():
            issues.append("CRITICAL: Gold features file missing — pipeline needs rerun")
            state["data_health"] = {"status": "FAILED"}
            state["data_issues"] = issues
            state["ingestion_status"] = "failed"
            return state

        df = pd.read_parquet(gold_path)
        health["total_rows"] = len(df)
        health["seasons"] = sorted(df["season"].unique().tolist()) if "season" in df.columns else []
        health["columns"] = len(df.columns)

        # Check 1: Null percentage per column
        null_pcts = (df.isnull().sum() / len(df) * 100).round(1)
        high_null_cols = null_pcts[null_pcts > 50].to_dict()
        if high_null_cols:
            issues.append(f"HIGH NULL RATE: {high_null_cols}")
        health["high_null_columns"] = high_null_cols

        # Check 2: Expected row count (should be ~20 drivers × ~22 races × 8 seasons)
        expected_min = 2500
        if len(df) < expected_min:
            issues.append(f"LOW ROW COUNT: {len(df)} rows (expected >{expected_min})")
        health["row_count_ok"] = len(df) >= expected_min

        # Check 3: Latest season check
        if "season" in df.columns:
            latest_season = int(df["season"].max())
            current_year = datetime.now().year
            if latest_season < current_year - 1:
                issues.append(f"STALE DATA: Latest season is {latest_season}, "
                              f"expected {current_year} or {current_year - 1}")
            health["latest_season"] = latest_season

        # Check 4: Target variable completeness
        for target in ["target_position", "target_podium", "target_winner"]:
            if target in df.columns:
                valid_pct = (df[target].notna().sum() / len(df) * 100)
                health[f"{target}_valid_pct"] = round(valid_pct, 1)
                if valid_pct < 80:
                    issues.append(f"LOW TARGET COVERAGE: {target} only {valid_pct:.0f}% valid")

        # Check 5: Feature distribution anomalies
        if "grid" in df.columns:
            grid_range = (int(df["grid"].min()), int(df["grid"].max()))
            if grid_range[1] > 25:
                issues.append(f"ANOMALY: Grid position range {grid_range} — max should be ~20")
            health["grid_range"] = grid_range

        health["status"] = "HEALTHY" if not issues else "ISSUES_FOUND"
        health["issues_count"] = len(issues)

    except Exception as e:
        issues.append(f"DATA AGENT ERROR: {str(e)}")
        health["status"] = "ERROR"

    state["data_health"] = health
    state["data_issues"] = issues
    state["ingestion_status"] = "ok" if not issues else "issues_found"

    # Log results
    log_path = AGENTS_LOG / f"data_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({"health": health, "issues": issues}, f, indent=2, default=str)

    logger.info(f"  Status: {health['status']} | Issues: {len(issues)}")
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")

    return state


# ═══════════════════════════════════════════════════════════
# Agent 2: Feature Agent — Propose, Test, Accept/Reject
# ═══════════════════════════════════════════════════════════

def feature_agent(state: AgentState) -> AgentState:
    """
    Autonomously engineers new features and tests their predictive value.
    This mimics a data scientist's hypothesis-test cycle:
      1. Analyze correlations in existing data
      2. Propose new derived features
      3. Test each feature's correlation with target
      4. Accept features that improve prediction
    """
    logger.info("🧪 FEATURE AGENT: Proposing and testing new features...")
    proposed = []
    accepted = []
    test_results = {}

    try:
        
        df = pd.read_parquet(GOLD_DIR / "race_prediction_features_complete.parquet")

        # ── Proposal 1: Constructor dominance index ──
        # Ratio of constructor points to field average
        if "constructor_rolling_points" in df.columns:
            field_avg = df.groupby(["season", "round"])["constructor_rolling_points"].transform("mean")
            df["constructor_dominance"] = df["constructor_rolling_points"] / field_avg.replace(0, np.nan)
            proposed.append({
                "name": "constructor_dominance",
                "description": "Ratio of constructor points to field average",
                "hypothesis": "Dominant constructors (ratio > 1.5) should predict podiums"
            })

            if "target_podium" in df.columns:
                valid = df[["constructor_dominance", "target_podium"]].dropna()
                if len(valid) > 100:
                    corr = valid["constructor_dominance"].corr(valid["target_podium"])
                    test_results["constructor_dominance"] = {
                        "correlation": round(corr, 4),
                        "accepted": abs(corr) > 0.1
                    }
                    if abs(corr) > 0.1:
                        accepted.append("constructor_dominance")
                        logger.info(f"  ✓ constructor_dominance: corr={corr:.3f} — ACCEPTED")
                    else:
                        logger.info(f"  ✗ constructor_dominance: corr={corr:.3f} — REJECTED")

        # ── Proposal 2: Qualifying improvement trend ──
        # Is the driver qualifying better or worse over recent races?
        if "quali_gap_pct" in df.columns:
            df_sorted = df.sort_values(["driver_id", "season", "round"])
            df["quali_trend"] = df_sorted.groupby("driver_id")["quali_gap_pct"].transform(
                lambda x: x.rolling(5, min_periods=2).apply(
                    lambda vals: np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) >= 2 else 0,
                    raw=False
                )
            )
            proposed.append({
                "name": "quali_trend",
                "description": "Slope of qualifying gap over last 5 races (negative = improving)",
                "hypothesis": "Drivers with improving qualifying trend should finish higher"
            })

            if "target_position" in df.columns:
                valid = df[["quali_trend", "target_position"]].dropna()
                valid = valid[np.isfinite(valid["quali_trend"])]
                if len(valid) > 100:
                    corr = valid["quali_trend"].corr(valid["target_position"])
                    test_results["quali_trend"] = {
                        "correlation": round(corr, 4),
                        "accepted": abs(corr) > 0.05
                    }
                    if abs(corr) > 0.05:
                        accepted.append("quali_trend")
                        logger.info(f"  ✓ quali_trend: corr={corr:.3f} — ACCEPTED")
                    else:
                        logger.info(f"  ✗ quali_trend: corr={corr:.3f} — REJECTED")

        # ── Proposal 3: Home race advantage ──
        # Does a driver perform better at their "home" circuits?
        if all(c in df.columns for c in ["circuit_avg_position", "rolling_avg_position_5"]):
            df["home_advantage"] = df["rolling_avg_position_5"] - df["circuit_avg_position"]
            proposed.append({
                "name": "home_advantage",
                "description": "Difference between overall form and circuit-specific form",
                "hypothesis": "Positive value = driver performs better at this circuit than average"
            })

            if "target_position" in df.columns:
                valid = df[["home_advantage", "target_position"]].dropna()
                if len(valid) > 100:
                    corr = valid["home_advantage"].corr(valid["target_position"])
                    test_results["home_advantage"] = {
                        "correlation": round(corr, 4),
                        "accepted": abs(corr) > 0.05
                    }
                    if abs(corr) > 0.05:
                        accepted.append("home_advantage")
                        logger.info(f"  ✓ home_advantage: corr={corr:.3f} — ACCEPTED")
                    else:
                        logger.info(f"  ✗ home_advantage: corr={corr:.3f} — REJECTED")

        # ── Proposal 4: Grid volatility ──
        # How much does this driver's grid position vary? Consistent qualifiers vs erratic.
        if "grid" in df.columns:
            df_sorted = df.sort_values(["driver_id", "season", "round"])
            df["grid_volatility"] = df_sorted.groupby("driver_id")["grid"].transform(
                lambda x: x.rolling(5, min_periods=2).std()
            )
            proposed.append({
                "name": "grid_volatility",
                "description": "Std dev of grid position over last 5 races",
                "hypothesis": "More consistent qualifiers should finish higher"
            })

            if "target_position" in df.columns:
                valid = df[["grid_volatility", "target_position"]].dropna()
                if len(valid) > 100:
                    corr = valid["grid_volatility"].corr(valid["target_position"])
                    test_results["grid_volatility"] = {
                        "correlation": round(corr, 4),
                        "accepted": abs(corr) > 0.05
                    }
                    if abs(corr) > 0.05:
                        accepted.append("grid_volatility")
                        logger.info(f"  ✓ grid_volatility: corr={corr:.3f} — ACCEPTED")
                    else:
                        logger.info(f"  ✗ grid_volatility: corr={corr:.3f} — REJECTED")

        # ── Proposal 5: Points momentum (acceleration) ──
        # Is the driver gaining points faster or slower?
        if "season_cumulative_points" in df.columns:
            df_sorted = df.sort_values(["driver_id", "season", "round"])
            df["points_acceleration"] = df_sorted.groupby(["driver_id", "season"])[
                "season_cumulative_points"
            ].diff().diff()
            proposed.append({
                "name": "points_acceleration",
                "description": "Second derivative of cumulative points — acceleration of form",
                "hypothesis": "Drivers gaining momentum faster should be more competitive"
            })

            if "target_podium" in df.columns:
                valid = df[["points_acceleration", "target_podium"]].dropna()
                valid = valid[np.isfinite(valid["points_acceleration"])]
                if len(valid) > 100:
                    corr = valid["points_acceleration"].corr(valid["target_podium"])
                    test_results["points_acceleration"] = {
                        "correlation": round(corr, 4),
                        "accepted": abs(corr) > 0.05
                    }
                    if abs(corr) > 0.05:
                        accepted.append("points_acceleration")
                        logger.info(f"  ✓ points_acceleration: corr={corr:.3f} — ACCEPTED")
                    else:
                        logger.info(f"  ✗ points_acceleration: corr={corr:.3f} — REJECTED")

    except Exception as e:
        logger.error(f"  Feature Agent error: {e}")
        state["errors"] = state.get("errors", []) + [str(e)]

    state["new_features_proposed"] = proposed
    state["new_features_accepted"] = accepted
    state["feature_test_results"] = test_results

    # Log
    log_path = AGENTS_LOG / f"feature_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({
            "proposed": proposed,
            "accepted": accepted,
            "test_results": test_results
        }, f, indent=2, default=str)

    logger.info(f"  Proposed: {len(proposed)} | Accepted: {len(accepted)}")
    return state


# ═══════════════════════════════════════════════════════════
# Agent 3: Model Agent — Evaluate, Diagnose, Improve
# ═══════════════════════════════════════════════════════════

def model_agent(state: AgentState) -> AgentState:
    """
    Evaluates current model performance and diagnoses failures.
    - Loads trained models and evaluates on latest data
    - Identifies where the model fails (specific circuits, conditions)
    - Recommends improvements
    - Triggers retraining if performance drops
    """
    logger.info("🤖 MODEL AGENT: Evaluating model performance...")
    performance = {}
    recommendations = []
    retrain = False

    try:
        import joblib

        df = pd.read_parquet(GOLD_DIR / "race_prediction_features.parquet")

        # Load models
        models = {}
        for name in ["position_predictor", "podium_classifier", "winner_predictor"]:
            path = MODELS_DIR / f"{name}.pkl"
            if path.exists():
                models[name] = joblib.load(path)
            else:
                recommendations.append(f"Model {name} not found — needs training")

        if not models:
            state["model_performance"] = {"status": "NO_MODELS"}
            state["model_recommendations"] = ["Run ml/train_models.py first"]
            state["retrain_triggered"] = False
            return state

        # Prepare features (same as training)
        from sklearn.preprocessing import LabelEncoder

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

        # Evaluate on 2025 data (most recent)
        test_mask = df["season"] >= 2025 if "season" in df.columns else pd.Series([False] * len(df))

        if test_mask.sum() > 0:
            X_test = X[test_mask]

            # Position predictor
            if "position_predictor" in models:
                y_true = df.loc[test_mask, "target_position"].dropna()
                valid_idx = y_true.index
                if len(valid_idx) > 0:
                    y_pred = models["position_predictor"].predict(X_test.loc[valid_idx])
                    mae = mean_absolute_error(y_true, y_pred)
                    performance["position_predictor"] = {
                        "mae_2025": round(mae, 2),
                        "test_rows": len(valid_idx)
                    }
                    if mae > 4.0:
                        recommendations.append(
                            f"Position predictor MAE={mae:.1f} on 2025 data — "
                            f"above threshold of 4.0, retrain recommended"
                        )
                        retrain = True
                    logger.info(f"  Position predictor MAE (2025): {mae:.2f}")

            # Podium classifier
            if "podium_classifier" in models:
                y_true = df.loc[test_mask, "target_podium"].dropna()
                valid_idx = y_true.index
                if len(valid_idx) > 0:
                    y_pred = models["podium_classifier"].predict(X_test.loc[valid_idx])
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    performance["podium_classifier"] = {
                        "f1_2025": round(f1, 3),
                        "test_rows": len(valid_idx)
                    }
                    if f1 < 0.5:
                        recommendations.append(
                            f"Podium classifier F1={f1:.2f} on 2025 — "
                            f"below threshold 0.5, retrain recommended"
                        )
                        retrain = True
                    logger.info(f"  Podium classifier F1 (2025): {f1:.3f}")

            # ── Failure analysis: where does the model struggle? ──
            if "position_predictor" in models and "circuit_name" in df.columns:
                test_df = df[test_mask].copy()
                y_true = test_df["target_position"]
                valid_mask = y_true.notna()
                if valid_mask.sum() > 0:
                    test_df = test_df[valid_mask]
                    test_df["pred_position"] = models["position_predictor"].predict(
                        X_test.loc[valid_mask.index[valid_mask]]
                    )
                    test_df["abs_error"] = abs(test_df["target_position"] - test_df["pred_position"])

                    # Worst circuits
                    circuit_errors = test_df.groupby("circuit_name")["abs_error"].mean().sort_values(ascending=False)
                    worst_circuits = circuit_errors.head(3).to_dict()
                    performance["worst_circuits_2025"] = {k: round(v, 1) for k, v in worst_circuits.items()}

                    if worst_circuits:
                        worst = list(worst_circuits.items())[0]
                        recommendations.append(
                            f"Model struggles most at {worst[0]} (avg error: {worst[1]:.1f} positions). "
                            f"Consider adding circuit-specific features like elevation, corners count."
                        )

                    # Worst drivers to predict
                    driver_errors = test_df.groupby("driver_name")["abs_error"].mean().sort_values(ascending=False)
                    worst_drivers = driver_errors.head(3).to_dict()
                    performance["hardest_to_predict_2025"] = {k: round(v, 1) for k, v in worst_drivers.items()}

        performance["status"] = "EVALUATED"

    except Exception as e:
        logger.error(f"  Model Agent error: {e}")
        performance["status"] = "ERROR"
        performance["error"] = str(e)

    state["model_performance"] = performance
    state["model_recommendations"] = recommendations
    state["retrain_triggered"] = retrain

    # Log
    log_path = AGENTS_LOG / f"model_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({
            "performance": performance,
            "recommendations": recommendations,
            "retrain_triggered": retrain
        }, f, indent=2, default=str)

    logger.info(f"  Recommendations: {len(recommendations)} | Retrain: {retrain}")
    return state


# ═══════════════════════════════════════════════════════════
# Agent 4: Insight Agent — Strategy Briefing Generator
# ═══════════════════════════════════════════════════════════
from llm_insight_agent import get_llm_briefing

def insight_agent(state: AgentState) -> AgentState:
    """
    Generates a natural-language race strategy briefing.
    Combines data health, feature insights, model predictions,
    and performance analysis into an actionable report.
    """
    logger.info("📊 INSIGHT AGENT: Generating race briefing...")

    try:
        df = pd.read_parquet(GOLD_DIR / "race_prediction_features.parquet")

        # Get latest race predictions
        if "season" in df.columns:
            latest_season = int(df["season"].max())
            latest_round = int(df[df["season"] == latest_season]["round"].max())
            latest_race = df[(df["season"] == latest_season) & (df["round"] == latest_round)]
            race_name = latest_race["race_name"].iloc[0] if "race_name" in latest_race.columns else "Unknown"
        else:
            latest_race = df.tail(20)
            race_name = "Unknown"
            latest_season = 0
            latest_round = 0

        # Build predictions dict
        predictions = {}
        if "target_position" in latest_race.columns and "driver_name" in latest_race.columns:
            results = latest_race.nsmallest(10, "target_position")[["driver_name", "target_position", "grid"]].to_dict("records")
            predictions["latest_results"] = results

        # ── Build the briefing ──
        briefing_parts = []

        briefing_parts.append(f"F1 STRATEGY BRIEFING — {race_name} ({latest_season} Round {latest_round})")
        briefing_parts.append("=" * 60)

        # Data health summary
        health = state.get("data_health", {})
        briefing_parts.append(f"\n[DATA STATUS] {health.get('status', 'UNKNOWN')}")
        briefing_parts.append(f"  Total records: {health.get('total_rows', 'N/A')}")
        briefing_parts.append(f"  Seasons covered: {health.get('seasons', 'N/A')}")
        if state.get("data_issues"):
            for issue in state["data_issues"][:3]:
                briefing_parts.append(f"  ⚠ {issue}")

        # Feature insights
        accepted_features = state.get("new_features_accepted", [])
        if accepted_features:
            briefing_parts.append(f"\n[NEW FEATURES DISCOVERED]")
            for feat in accepted_features:
                results = state.get("feature_test_results", {}).get(feat, {})
                corr = results.get("correlation", "N/A")
                briefing_parts.append(f"  + {feat}: correlation with target = {corr}")

        # Model performance
        perf = state.get("model_performance", {})
        briefing_parts.append(f"\n[MODEL PERFORMANCE]")
        if "position_predictor" in perf:
            briefing_parts.append(f"  Position predictor MAE: {perf['position_predictor'].get('mae_2025', 'N/A')}")
        if "podium_classifier" in perf:
            briefing_parts.append(f"  Podium classifier F1: {perf['podium_classifier'].get('f1_2025', 'N/A')}")
        if "worst_circuits_2025" in perf:
            briefing_parts.append(f"  Hardest circuits: {perf['worst_circuits_2025']}")
        if "hardest_to_predict_2025" in perf:
            briefing_parts.append(f"  Hardest to predict: {perf['hardest_to_predict_2025']}")

        # Recommendations
        recs = state.get("model_recommendations", [])
        if recs:
            briefing_parts.append(f"\n[RECOMMENDATIONS]")
            for i, rec in enumerate(recs, 1):
                briefing_parts.append(f"  {i}. {rec}")

        # Latest race results
        if predictions.get("latest_results"):
            briefing_parts.append(f"\n[LATEST RACE RESULTS — {race_name}]")
            for r in predictions["latest_results"]:
                briefing_parts.append(
                    f"  P{int(r.get('target_position', 0)):2d} | "
                    f"Grid {int(r.get('grid', 0)):2d} | "
                    f"{r.get('driver_name', 'Unknown')}"
                )

        briefing_parts.append(f"\n{'=' * 60}")
        briefing_parts.append(f"Generated: {datetime.now().isoformat()}")
        briefing_parts.append(f"Retrain triggered: {state.get('retrain_triggered', False)}")

        

        briefing = "\n".join(briefing_parts)

        # ── Generate LLM-enhanced briefing ──
        try:
            llm_briefing = get_llm_briefing(state)
            briefing = briefing + "\n\n" + "=" * 60 + "\n[AI-POWERED ANALYSIS — Groq LLaMA 3.1]\n" + "=" * 60 + "\n\n" + llm_briefing
            
            logger.info("  ✅ LLM briefing generated successfully")
        except Exception as llm_err:
            logger.warning(f"  ⚠️ LLM briefing skipped: {llm_err}")

        # Save briefing
        briefing_path = AGENTS_LOG / f"briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(briefing_path, "w") as f:
            f.write(briefing)

        logger.info(f"  Briefing generated: {briefing_path}")
        print("\n" + briefing)

    except Exception as e:
        briefing = f"Insight Agent error: {e}"
        predictions = {}
        logger.error(f"  {e}")

    state["race_briefing"] = briefing
    state["predictions"] = predictions
    return state


# ═══════════════════════════════════════════════════════════
# LangGraph Orchestrator — Coordinates All Agents
# ═══════════════════════════════════════════════════════════

def should_retrain(state: AgentState) -> Literal["retrain", "skip"]:
    """Decision node: should we trigger model retraining?"""
    if state.get("retrain_triggered", False):
        return "retrain"
    return "skip"


def retrain_models(state: AgentState) -> AgentState:
    """Retrain ML models when performance drops."""
    logger.info("🔄 ORCHESTRATOR: Triggering model retraining...")
    try:
        # Import and run training pipeline
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from ml.train_models import main as train_main
        train_main()
        logger.info("  Retraining complete ✓")
    except Exception as e:
        logger.error(f"  Retraining failed: {e}")
        state["errors"] = state.get("errors", []) + [f"Retrain failed: {e}"]
    return state


def build_agent_graph():
    """Build the LangGraph agent orchestration pipeline."""
    graph = StateGraph(AgentState)

    # Add agent nodes
    graph.add_node("data_agent", data_agent)
    graph.add_node("feature_agent", feature_agent)
    graph.add_node("model_agent", model_agent)
    graph.add_node("insight_agent", insight_agent)
    graph.add_node("retrain_models", retrain_models)

    # Define flow: data → feature → model → (retrain?) → insight
    graph.set_entry_point("data_agent")
    graph.add_edge("data_agent", "feature_agent")
    graph.add_edge("feature_agent", "model_agent")

    # Conditional: retrain or skip to insights
    graph.add_conditional_edges(
        "model_agent",
        should_retrain,
        {
            "retrain": "retrain_models",
            "skip": "insight_agent",
        }
    )
    graph.add_edge("retrain_models", "insight_agent")
    graph.add_edge("insight_agent", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    logger.info("🏎️  F1 Agentic AI Pipeline — Phase 3")
    logger.info("=" * 60)

    # Build and run the agent graph
    pipeline = build_agent_graph()

    # Initialize state
    initial_state: AgentState = {
        "data_health": {},
        "ingestion_status": "",
        "data_issues": [],
        "new_features_proposed": [],
        "new_features_accepted": [],
        "feature_test_results": {},
        "model_performance": {},
        "model_recommendations": [],
        "retrain_triggered": False,
        "race_briefing": "",
        "predictions": {},
        "phase": "starting",
        "errors": [],
        "timestamp": datetime.now().isoformat(),
    }
    

    # Load memory from past runs
    past_runs = load_memory()
    initial_state["memory_context"] = get_memory_context(past_runs)
    logger.info(f"📚 Loaded {len(past_runs)} past agent runs from memory")
    # Run the pipeline
    final_state = pipeline.invoke(initial_state)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("AGENT PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data Health:     {final_state['data_health'].get('status', 'N/A')}")
    logger.info(f"Features Found:  {len(final_state['new_features_accepted'])} accepted / "
                f"{len(final_state['new_features_proposed'])} proposed")
    logger.info(f"Model Status:    {final_state['model_performance'].get('status', 'N/A')}")
    logger.info(f"Retrain:         {final_state['retrain_triggered']}")
    logger.info(f"Errors:          {len(final_state['errors'])}")
    logger.info("=" * 60)
    logger.info("🏁 Agentic pipeline complete")
    
    # Save this run to agent memory
    save_agent_memory(final_state)
    # Save full state
    state_path = AGENTS_LOG / f"pipeline_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(state_path, "w") as f:
        json.dump(final_state, f, indent=2, default=str)


if __name__ == "__main__":
    main()
