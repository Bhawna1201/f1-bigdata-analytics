"""
F1 Big Data — Enhanced Pipeline
=================================
Fixes 3 gaps:
1. Integrates weather data into ML features
2. Streams 2026 live season data
3. Demonstrates scheduled job execution

Run: python enhanced_pipeline.py
"""
import time
import logging
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger("enhanced_pipeline")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# PART 1: Weather Data Integration
# ═══════════════════════════════════════════════════════════

def build_weather_features():
    """
    Integrate weather data into the ML pipeline.

    WHY WEATHER MATTERS IN F1:
    - Rain completely changes race outcomes (favorites crash, underdogs win)
    - Track temperature affects tire degradation (hot = faster wear)
    - Air temperature affects engine cooling and aerodynamics
    - Wind affects car balance especially at high-speed circuits
    - Humidity affects tire grip levels

    WHAT WE BUILD:
    - avg_track_temp: Average track temperature during the race
    - avg_air_temp: Average air temperature
    - max_humidity: Maximum humidity (higher = more rain risk)
    - had_rain: Binary flag — did it rain during the race?
    - temp_variation: Temperature range during race (stability indicator)
    - is_hot_race: Track temp > 40°C (extreme tire wear expected)
    - is_cold_race: Track temp < 25°C (grip issues, tire warm-up problems)
    """
    logger.info("🌦️  Building weather features...")

    weather_dir = BRONZE_DIR / "fastf1"
    weather_files = sorted(weather_dir.glob("weather_*.parquet"))

    if not weather_files:
        logger.warning("No weather files found — skipping weather features")
        return None

    all_weather = []
    for f in weather_files:
        try:
            df = pd.read_parquet(f)
            all_weather.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f.name}: {e}")

    if not all_weather:
        return None

    weather = pd.concat(all_weather, ignore_index=True)
    logger.info(f"Loaded {len(weather)} weather records from {len(weather_files)} files")

    # Identify temperature columns (FastF1 uses different names across versions)
    track_temp_col = None
    air_temp_col = None
    humidity_col = None
    rainfall_col = None

    for col in weather.columns:
        col_lower = col.lower()
        if "tracktemp" in col_lower or "track_temp" in col_lower:
            track_temp_col = col
        elif "airtemp" in col_lower or "air_temp" in col_lower:
            air_temp_col = col
        elif "humidity" in col_lower:
            humidity_col = col
        elif "rainfall" in col_lower or "rain" in col_lower:
            rainfall_col = col

    logger.info(f"Weather columns found: track_temp={track_temp_col}, "
                f"air_temp={air_temp_col}, humidity={humidity_col}, rainfall={rainfall_col}")

    # Aggregate weather per race (season + round)
    agg_dict = {}
    if track_temp_col:
        weather[track_temp_col] = pd.to_numeric(weather[track_temp_col], errors="coerce")
        agg_dict["avg_track_temp"] = (track_temp_col, "mean")
        agg_dict["max_track_temp"] = (track_temp_col, "max")
        agg_dict["min_track_temp"] = (track_temp_col, "min")

    if air_temp_col:
        weather[air_temp_col] = pd.to_numeric(weather[air_temp_col], errors="coerce")
        agg_dict["avg_air_temp"] = (air_temp_col, "mean")

    if humidity_col:
        weather[humidity_col] = pd.to_numeric(weather[humidity_col], errors="coerce")
        agg_dict["max_humidity"] = (humidity_col, "max")
        agg_dict["avg_humidity"] = (humidity_col, "mean")

    if rainfall_col:
        weather[rainfall_col] = pd.to_numeric(weather[rainfall_col], errors="coerce")
        agg_dict["had_rain_raw"] = (rainfall_col, "max")

    if not agg_dict:
        logger.warning("No usable weather columns found")
        return None

    weather_agg = weather.groupby(["season", "round"]).agg(**agg_dict).reset_index()

    # Derived weather features
    if "avg_track_temp" in weather_agg.columns:
        weather_agg["is_hot_race"] = (weather_agg["avg_track_temp"] > 40).astype(int)
        weather_agg["is_cold_race"] = (weather_agg["avg_track_temp"] < 25).astype(int)

    if "max_track_temp" in weather_agg.columns and "min_track_temp" in weather_agg.columns:
        weather_agg["temp_variation"] = weather_agg["max_track_temp"] - weather_agg["min_track_temp"]

    if "had_rain_raw" in weather_agg.columns:
        weather_agg["had_rain"] = (weather_agg["had_rain_raw"] > 0).astype(int)
        weather_agg = weather_agg.drop(columns=["had_rain_raw"])

    # Save weather features
    path = SILVER_DIR / "weather_features.parquet"
    weather_agg.to_parquet(path, compression="snappy", index=False)
    logger.info(f"✓ Weather features: {len(weather_agg)} race-weather records → {path}")
    logger.info(f"  Columns: {list(weather_agg.columns)}")

    return weather_agg


def merge_weather_into_gold():
    """
    Merge weather features into the Gold ML feature table.
    This adds 5-8 new weather columns to each race-driver record.
    """
    logger.info("🔗 Merging weather into Gold features...")

    gold_path = GOLD_DIR / "race_prediction_features.parquet"
    weather_path = SILVER_DIR / "weather_features.parquet"

    if not gold_path.exists() or not weather_path.exists():
        logger.warning("Gold features or weather features missing — skipping merge")
        return None

    features = pd.read_parquet(gold_path)
    weather = pd.read_parquet(weather_path)

    # Drop any existing weather columns to avoid duplicates
    weather_cols = [c for c in weather.columns if c not in ["season", "round"]]
    features = features.drop(columns=[c for c in weather_cols if c in features.columns], errors="ignore")

    # Merge
    enhanced = features.merge(weather, on=["season", "round"], how="left")

    # Save enhanced features
    enhanced_path = GOLD_DIR / "race_prediction_features_weather.parquet"
    enhanced.to_parquet(enhanced_path, compression="snappy", index=False)

    new_cols = len(enhanced.columns) - len(features.columns)
    weather_coverage = enhanced[weather_cols[0]].notna().mean() * 100 if weather_cols else 0

    logger.info(f"✓ Enhanced features: {len(enhanced)} rows, {len(enhanced.columns)} columns "
                f"(+{new_cols} weather features)")
    logger.info(f"  Weather coverage: {weather_coverage:.0f}% of races have weather data")

    return enhanced


# ═══════════════════════════════════════════════════════════
# PART 2: 2026 Live Season Data Streaming
# ═══════════════════════════════════════════════════════════

def ingest_2026_season():
    """
    Pull 2026 season data (live/current season).

    HOW THIS WORKS:
    - Jolpica API provides results within hours of race finish
    - FastF1 provides telemetry within 1-2 days
    - We check which rounds have happened and pull them
    - This runs as a scheduled job every Monday via Airflow
    """
    logger.info("🏎️  Ingesting 2026 season data...")

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        import fastf1
        from ingestion.bronze_ingestion import ErgastIngestion, FastF1Ingestion

        # Check how many 2026 races have data
        try:
            import requests
            resp = requests.get("https://api.jolpi.ca/ergast/f1/2026/results.json?limit=5", timeout=15)
            if resp.status_code == 200:
                total = int(resp.json()["MRData"]["total"])
                if total > 0:
                    logger.info(f"  Jolpica has {total} race results for 2026")

                    ergast = ErgastIngestion()
                    ergast.ingest_race_results(2026)
                    time.sleep(3)
                    ergast.ingest_qualifying(2026)
                    time.sleep(3)
                    ergast.ingest_pit_stops(2026)
                    time.sleep(3)
                    ergast.ingest_driver_standings(2026)
                    logger.info("  ✓ Ergast 2026 data ingested")
                else:
                    logger.info("  No 2026 race results available yet")
            else:
                logger.info(f"  Jolpica returned {resp.status_code} for 2026 — season may not have started")
        except Exception as e:
            logger.warning(f"  Ergast 2026 error: {e}")

        # FastF1 for 2026
        try:
            schedule = fastf1.get_event_schedule(2026)
            past_events = schedule[schedule["EventDate"] < datetime.now().strftime("%Y-%m-%d")]
            race_rounds = past_events[past_events["EventFormat"] != "testing"]["RoundNumber"].tolist()

            if race_rounds:
                logger.info(f"  FastF1 has {len(race_rounds)} completed 2026 rounds")
                ff1 = FastF1Ingestion()
                for rnd in race_rounds:
                    if rnd == 0:
                        continue
                    try:
                        ff1.ingest_session_laps(2026, rnd, "R")
                        ff1.ingest_weather(2026, rnd, "R")
                        time.sleep(3)
                    except Exception as e:
                        if "RateLimit" in str(e) or "500 calls" in str(e):
                            logger.warning(f"  Rate limited at R{rnd} — will retry next run")
                            break
                        logger.warning(f"  FastF1 2026 R{rnd}: {e}")
                logger.info("  ✓ FastF1 2026 data ingested")
            else:
                logger.info("  No completed 2026 FastF1 events yet")
        except Exception as e:
            logger.warning(f"  FastF1 2026 schedule error: {e}")

    except ImportError as e:
        logger.error(f"  Import error: {e}")


# ═══════════════════════════════════════════════════════════
# PART 3: Demonstrate Scheduled Job Execution
# ═══════════════════════════════════════════════════════════

def run_scheduled_pipeline():
    """
    This is what Airflow runs every Monday after a race weekend.
    Demonstrates the full automated cycle:

    1. Check for new race data → ingest
    2. Rebuild Silver layer (clean + join)
    3. Rebuild Gold layer (features + weather)
    4. Retrain ML models if needed
    5. Run Agentic AI pipeline
    6. Log execution for audit trail
    """
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_log = {
        "job_id": job_id,
        "started_at": datetime.now().isoformat(),
        "steps": [],
        "status": "running"
    }

    logger.info(f"📅 SCHEDULED JOB {job_id} — Starting automated pipeline")
    logger.info("=" * 60)

    # Step 1: Ingest latest data
    logger.info("STEP 1/5: Ingesting latest race data...")
    try:
        ingest_2026_season()
        job_log["steps"].append({"step": "ingest", "status": "ok", "time": datetime.now().isoformat()})
    except Exception as e:
        job_log["steps"].append({"step": "ingest", "status": "error", "error": str(e)})
        logger.error(f"  Ingestion failed: {e}")

    # Step 2: Rebuild Silver + Gold
    logger.info("\nSTEP 2/5: Rebuilding Silver layer...")
    try:
        from ingestion.silver_transform import SilverTransformPipeline
        SilverTransformPipeline().run_all()
        job_log["steps"].append({"step": "silver", "status": "ok", "time": datetime.now().isoformat()})
    except Exception as e:
        job_log["steps"].append({"step": "silver", "status": "error", "error": str(e)})
        logger.error(f"  Silver transform failed: {e}")

    logger.info("\nSTEP 3/5: Rebuilding Gold features + weather...")
    try:
        from ingestion.gold_features import GoldFeaturePipeline
        GoldFeaturePipeline().run_all()

        # Add weather features
        build_weather_features()
        merge_weather_into_gold()
        job_log["steps"].append({"step": "gold_weather", "status": "ok", "time": datetime.now().isoformat()})
    except Exception as e:
        job_log["steps"].append({"step": "gold_weather", "status": "error", "error": str(e)})
        logger.error(f"  Gold/weather failed: {e}")

    # Step 4: Check if retrain needed
    logger.info("\nSTEP 4/5: Checking model performance...")
    try:
        import joblib
        from sklearn.metrics import mean_absolute_error

        model_path = PROJECT_ROOT / "models" / "position_predictor.pkl"
        if model_path.exists():
            features = pd.read_parquet(GOLD_DIR / "race_prediction_features.parquet")
            # Quick performance check on latest season
            latest_season = features["season"].max()
            latest = features[features["season"] == latest_season]

            if len(latest) > 20:
                from sklearn.preprocessing import LabelEncoder
                cat_cols = ["driver_id", "driver_code", "driver_name",
                            "constructor_id", "constructor_name",
                            "circuit_id", "circuit_name", "race_name"]
                for col in cat_cols:
                    if col in features.columns:
                        le = LabelEncoder()
                        features[col + "_enc"] = le.fit_transform(features[col].astype(str))

                drop_cols = cat_cols + ["race_date", "target_position", "target_podium",
                                        "target_points_finish", "target_winner", "position"]
                feature_cols = [c for c in features.columns if c not in drop_cols
                                and features[c].dtype in ["float64", "int64", "int32", "float32"]]
                X = features[feature_cols].fillna(features[feature_cols].median())
                X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

                model = joblib.load(model_path)
                latest_mask = features["season"] == latest_season
                y_true = features.loc[latest_mask, "target_position"].dropna()
                valid_idx = y_true.index
                y_pred = model.predict(X.loc[valid_idx])
                mae = mean_absolute_error(y_true, y_pred)

                logger.info(f"  Current MAE on {latest_season} data: {mae:.2f}")

                if mae > 4.0:
                    logger.info("  MAE > 4.0 — triggering retraining...")
                    from ml.train_models import main as train_main
                    train_main()
                    job_log["steps"].append({"step": "retrain", "status": "triggered",
                                             "mae": round(mae, 2)})
                else:
                    logger.info(f"  MAE {mae:.2f} is acceptable — no retraining needed")
                    job_log["steps"].append({"step": "model_check", "status": "ok",
                                             "mae": round(mae, 2)})
        else:
            logger.info("  No model found — training from scratch...")
            from ml.train_models import main as train_main
            train_main()
            job_log["steps"].append({"step": "initial_train", "status": "ok"})

    except Exception as e:
        job_log["steps"].append({"step": "model_check", "status": "error", "error": str(e)})
        logger.error(f"  Model check failed: {e}")

    # Step 5: Run Agentic AI
    logger.info("\nSTEP 5/5: Running Agentic AI pipeline...")
    try:
        from agents.agentic_pipeline import build_agent_graph, AgentState
        pipeline = build_agent_graph()
        initial_state = {
            "data_health": {}, "ingestion_status": "", "data_issues": [],
            "new_features_proposed": [], "new_features_accepted": [],
            "feature_test_results": {}, "model_performance": {},
            "model_recommendations": [], "retrain_triggered": False,
            "race_briefing": "", "predictions": {},
            "phase": "scheduled_run", "errors": [],
            "timestamp": datetime.now().isoformat(),
        }
        final_state = pipeline.invoke(initial_state)
        job_log["steps"].append({"step": "agentic_ai", "status": "ok",
                                  "features_found": len(final_state.get("new_features_accepted", []))})
    except Exception as e:
        job_log["steps"].append({"step": "agentic_ai", "status": "error", "error": str(e)})
        logger.error(f"  Agentic AI failed: {e}")

    # Log the job execution
    job_log["completed_at"] = datetime.now().isoformat()
    job_log["status"] = "completed"
    successful_steps = sum(1 for s in job_log["steps"] if s["status"] in ("ok", "triggered"))
    job_log["success_rate"] = f"{successful_steps}/{len(job_log['steps'])}"

    log_path = LOGS_DIR / f"scheduled_job_{job_id}.json"
    with open(log_path, "w") as f:
        json.dump(job_log, f, indent=2, default=str)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SCHEDULED JOB {job_id} COMPLETE")
    logger.info(f"  Steps: {successful_steps}/{len(job_log['steps'])} successful")
    logger.info(f"  Log: {log_path}")
    logger.info(f"{'=' * 60}")

    return job_log


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    logger.info("F1 Big Data — Enhanced Pipeline")
    logger.info("=" * 60)
    logger.info("This script demonstrates:")
    logger.info("  1. Weather data integration into ML features")
    logger.info("  2. 2026 live season data streaming")
    logger.info("  3. Full scheduled job execution (what Airflow runs)")
    logger.info("=" * 60)

    # Run the full scheduled pipeline
    run_scheduled_pipeline()
