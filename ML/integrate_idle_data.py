"""
Integrate All Idle Data into ML Features
==========================================
Fixes 3 gaps where collected data wasn't being used:

1. driver_standings → championship_points_end, season_wins, season_position_prev
2. tire_degradation → avg_tire_deg_slope, soft_deg_rate, hard_deg_rate
3. laps (178K rows) → avg_lap_time, lap_consistency, best_lap_pct, stint_count

Usage: python ml/integrate_idle_data.py
"""
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BRONZE_DIR, SILVER_DIR, GOLD_DIR

logger = logging.getLogger("integrate_idle")


def integrate_standings():
    """
    Integrate driver_standings into ML features.

    WHAT THIS ADDS:
    - prev_season_position: Where did this driver finish in LAST year's championship?
    - prev_season_wins: How many wins did they have last season?
    - prev_season_points: Total points last season
    - is_champion: Did they win the championship last year? (1/0)
    - is_top3_championship: Were they top 3 last year?

    WHY IT MATTERS:
    - A driver who was champion last year starts the new season with confidence
    - Teams with more points get more prize money → better car development
    """
    logger.info("📊 Integrating driver_standings...")

    standings_files = sorted(BRONZE_DIR.glob("ergast/driver_standings_*.parquet"))
    if not standings_files:
        logger.warning("No standings files found")
        return None

    all_standings = []
    for f in standings_files:
        df = pd.read_parquet(f)
        all_standings.append(df)

    standings = pd.concat(all_standings, ignore_index=True)
    logger.info(f"  Loaded {len(standings)} standings records from {len(standings_files)} files")

    # Ensure numeric types
    for col in ["season", "position", "points", "wins"]:
        if col in standings.columns:
            standings[col] = pd.to_numeric(standings[col], errors="coerce")

    # Create previous season features (shift by 1 year)
    standings_prev = standings[["season", "driver_id", "position", "points", "wins"]].copy()
    standings_prev["season"] = standings_prev["season"] + 1  # Shift to next season
    standings_prev = standings_prev.rename(columns={
        "position": "prev_season_position",
        "points": "prev_season_points",
        "wins": "prev_season_wins"
    })

    standings_prev["is_prev_champion"] = (standings_prev["prev_season_position"] == 1).astype(int)
    standings_prev["is_prev_top3"] = (standings_prev["prev_season_position"] <= 3).astype(int)
    standings_prev["is_prev_top5"] = (standings_prev["prev_season_position"] <= 5).astype(int)

    logger.info(f"  ✓ Created prev_season features for {len(standings_prev)} driver-seasons")
    return standings_prev


def integrate_tire_degradation():
    """
    Integrate tire_degradation_features into ML features.

    WHAT THIS ADDS (per driver per race):
    - avg_tire_deg_slope: Average tire degradation rate across all stints
    - soft_deg_slope: Degradation rate on soft tires specifically
    - medium_deg_slope: Degradation on mediums
    - hard_deg_slope: Degradation on hards
    - driver_avg_deg: Driver's historical average degradation (are they easy on tires?)

    WHY IT MATTERS:
    - Drivers who manage tires better can do longer stints → fewer pit stops
    - Tire management is a key differentiator between good and great drivers
    - Some cars are harder on tires than others
    """
    logger.info("🛞 Integrating tire_degradation...")

    tire_path = GOLD_DIR / "tire_degradation_features.parquet"
    if not tire_path.exists():
        logger.warning("No tire degradation file found")
        return None

    tire = pd.read_parquet(tire_path)
    logger.info(f"  Loaded {len(tire)} tire degradation records")

    # Ensure numeric types
    for col in ["season", "round", "tire_deg_slope", "tire_deg_r2"]:
        if col in tire.columns:
            tire[col] = pd.to_numeric(tire[col], errors="coerce")

    # Filter reliable fits only (R² > 0.3)
    reliable = tire[tire["tire_deg_r2"] > 0.3].copy() if "tire_deg_r2" in tire.columns else tire.copy()

    driver_col = "Driver" if "Driver" in reliable.columns else "driver_id"

    # Aggregate per race per driver
    race_tire = reliable.groupby(["season", "round", driver_col]).agg(
        avg_tire_deg_slope=("tire_deg_slope", "mean"),
        max_tire_deg_slope=("tire_deg_slope", "max"),
        min_tire_deg_slope=("tire_deg_slope", "min"),
        stint_count_tire=("tire_deg_slope", "count"),
    ).reset_index()

    # Per compound degradation (if available)
    if "Compound" in reliable.columns:
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            comp_data = reliable[reliable["Compound"] == compound]
            if len(comp_data) > 0:
                comp_agg = comp_data.groupby(["season", "round", driver_col]).agg(
                    **{f"{compound.lower()}_deg_slope": ("tire_deg_slope", "mean")}
                ).reset_index()
                race_tire = race_tire.merge(comp_agg, on=["season", "round", driver_col], how="left")

    # Driver's historical average degradation (rolling — are they easy on tires?)
    race_tire = race_tire.sort_values(["season", "round"])
    race_tire["driver_avg_deg_5"] = race_tire.groupby(driver_col)["avg_tire_deg_slope"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )

    # Rename driver column to match Gold table
    if driver_col == "Driver":
        race_tire = race_tire.rename(columns={"Driver": "driver_code"})

    logger.info(f"  ✓ Tire features: {len(race_tire)} race-driver records")
    logger.info(f"  Columns: {list(race_tire.columns)}")
    return race_tire


def integrate_lap_features():
    """
    Aggregate 178K lap records into per-race features.

    WHAT THIS ADDS:
    - avg_lap_time: Driver's average lap time in the race
    - lap_consistency: Std dev of lap times (lower = more consistent)
    - best_lap_pct: Best lap as % of race best (pace indicator)
    - laps_completed: How many laps did they complete?
    - avg_sector_position: Average position during race (not just finish)
    - pace_vs_field: Average lap time vs field average (seconds)
    - num_stints: Number of tire stints
    - avg_stint_length: Average laps per stint

    WHY IT MATTERS:
    - Race pace is different from qualifying pace
    - Consistency matters — a driver with 0.5s std dev is more predictable
    - Stint length reveals strategy patterns
    """
    logger.info("⏱️  Integrating lap-level features (178K rows → per-race aggregates)...")

    laps_path = SILVER_DIR / "laps.parquet"
    if not laps_path.exists():
        logger.warning("No laps file found")
        return None

    laps = pd.read_parquet(laps_path)
    logger.info(f"  Loaded {len(laps)} lap records")

    # Ensure numeric
    for col in ["season", "round", "LapTime_seconds", "LapNumber", "TyreLife"]:
        if col in laps.columns:
            laps[col] = pd.to_numeric(laps[col], errors="coerce")

    driver_col = "Driver" if "Driver" in laps.columns else "driver_id"

    # Filter out outlier laps (pit in/out, safety car)
    if "LapTime_seconds" in laps.columns:
        race_median = laps.groupby(["season", "round"])["LapTime_seconds"].transform("median")
        clean_laps = laps[
            (laps["LapTime_seconds"].notna()) &
            (laps["LapTime_seconds"] > 0) &
            (laps["LapTime_seconds"] < race_median * 1.15)  # Within 15% of median
        ].copy()
    else:
        clean_laps = laps.copy()

    logger.info(f"  Clean laps (after removing outliers): {len(clean_laps)}")

    # Aggregate per driver per race
    agg_dict = {}
    if "LapTime_seconds" in clean_laps.columns:
        agg_dict["avg_lap_time"] = ("LapTime_seconds", "mean")
        agg_dict["lap_consistency"] = ("LapTime_seconds", "std")
        agg_dict["best_lap_time"] = ("LapTime_seconds", "min")

    if "LapNumber" in clean_laps.columns:
        agg_dict["laps_completed_actual"] = ("LapNumber", "max")

    if "TyreLife" in clean_laps.columns:
        agg_dict["max_tyre_life"] = ("TyreLife", "max")

    if not agg_dict:
        logger.warning("  No aggregatable columns found in laps")
        return None

    lap_features = clean_laps.groupby(["season", "round", driver_col]).agg(**agg_dict).reset_index()

    # Pace vs field average
    if "avg_lap_time" in lap_features.columns:
        field_avg = lap_features.groupby(["season", "round"])["avg_lap_time"].transform("mean")
        lap_features["pace_vs_field"] = lap_features["avg_lap_time"] - field_avg

        race_best = lap_features.groupby(["season", "round"])["best_lap_time"].transform("min")
        lap_features["best_lap_pct"] = (lap_features["best_lap_time"] / race_best - 1) * 100

    # Count stints (number of compound changes)
    if "Compound" in clean_laps.columns:
            stint_counts = clean_laps.groupby(["season", "round", driver_col])["Compound"].apply(
                lambda x: x.ne(x.shift()).sum()
            ).reset_index(name="num_stints")
            lap_features = lap_features.merge(stint_counts, on=["season", "round", driver_col], how="left")
        

    # Rolling driver consistency (historical — is this driver generally consistent?)
    if "lap_consistency" in lap_features.columns:
        lap_features = lap_features.sort_values(["season", "round"])
        lap_features["driver_consistency_5"] = lap_features.groupby(driver_col)[
            "lap_consistency"
        ].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))

    # Rename driver column
    if driver_col == "Driver":
        lap_features = lap_features.rename(columns={"Driver": "driver_code"})

    logger.info(f"  ✓ Lap features: {len(lap_features)} race-driver records")
    logger.info(f"  Columns: {list(lap_features.columns)}")
    return lap_features


def merge_all_into_gold():
    """Merge standings, tire, and lap features into the advanced Gold table."""
    logger.info("\n🔗 Merging all features into Gold table...")

    # Load current advanced features
    advanced_path = GOLD_DIR / "race_prediction_features_advanced.parquet"
    if not advanced_path.exists():
        advanced_path = GOLD_DIR / "race_prediction_features_weather.parquet"
    if not advanced_path.exists():
        advanced_path = GOLD_DIR / "race_prediction_features.parquet"

    gold = pd.read_parquet(advanced_path)
    original_cols = len(gold.columns)
    logger.info(f"  Base: {len(gold)} rows, {original_cols} columns")

    # 1. Merge standings
    standings = integrate_standings()
    if standings is not None:
        gold = gold.merge(
            standings, on=["season", "driver_id"], how="left"
        )
        # Fill missing (first season drivers)
        gold["prev_season_position"] = gold["prev_season_position"].fillna(20)
        gold["prev_season_points"] = gold["prev_season_points"].fillna(0)
        gold["prev_season_wins"] = gold["prev_season_wins"].fillna(0)
        gold["is_prev_champion"] = gold["is_prev_champion"].fillna(0).astype(int)
        gold["is_prev_top3"] = gold["is_prev_top3"].fillna(0).astype(int)
        gold["is_prev_top5"] = gold["is_prev_top5"].fillna(0).astype(int)

    # 2. Merge tire degradation
    tire = integrate_tire_degradation()
    if tire is not None:
        merge_key = ["season", "round"]
        if "driver_code" in tire.columns and "driver_code" in gold.columns:
            merge_key.append("driver_code")
        elif "driver_id" in tire.columns and "driver_id" in gold.columns:
            merge_key.append("driver_id")

        # Drop any existing tire columns to avoid dupes
        tire_new_cols = [c for c in tire.columns if c not in merge_key]
        gold = gold.drop(columns=[c for c in tire_new_cols if c in gold.columns], errors="ignore")
        gold = gold.merge(tire, on=merge_key, how="left")

    # 3. Merge lap features
    lap_feats = integrate_lap_features()
    if lap_feats is not None:
        merge_key = ["season", "round"]
        if "driver_code" in lap_feats.columns and "driver_code" in gold.columns:
            merge_key.append("driver_code")
        elif "driver_id" in lap_feats.columns and "driver_id" in gold.columns:
            merge_key.append("driver_id")

        lap_new_cols = [c for c in lap_feats.columns if c not in merge_key]
        gold = gold.drop(columns=[c for c in lap_new_cols if c in gold.columns], errors="ignore")
        gold = gold.merge(lap_feats, on=merge_key, how="left")

    # Save
    output_path = GOLD_DIR / "race_prediction_features_complete.parquet"
    gold.to_parquet(output_path, compression="snappy", index=False)

    new_cols = len(gold.columns) - original_cols
    logger.info(f"\n✅ COMPLETE Gold features: {len(gold)} rows, {len(gold.columns)} columns (+{new_cols} new)")

    # Summary of all features by category
    all_cols = gold.columns.tolist()
    logger.info(f"\n  Feature categories:")
    logger.info(f"    Base features (race/quali/pit):    ~29")
    logger.info(f"    Weather features:                  ~10")
    logger.info(f"    Advanced features (ELO/streaks):   ~20")
    logger.info(f"    Standings features:                ~{len([c for c in all_cols if 'prev_season' in c or 'champion' in c])}")
    logger.info(f"    Tire degradation features:         ~{len([c for c in all_cols if 'deg' in c or 'tire' in c or 'stint' in c.lower()])}")
    logger.info(f"    Lap-level features:                ~{len([c for c in all_cols if 'lap' in c.lower() or 'pace' in c or 'consistency' in c])}")
    logger.info(f"    TOTAL:                             {len(gold.columns)}")

    # Check what percentage of data is filled
    fill_rates = (gold.notna().sum() / len(gold) * 100).round(1)
    low_fill = fill_rates[fill_rates < 50]
    if len(low_fill) > 0:
        logger.info(f"\n  Low fill rate columns (<50%):")
        for col, rate in low_fill.items():
            logger.info(f"    {col}: {rate}%")

    return gold


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    logger.info("🏎️  Integrating ALL idle data into ML features")
    logger.info("=" * 60)

    gold = merge_all_into_gold()

    logger.info("\n" + "=" * 60)
    logger.info("NO MORE IDLE DATA — everything collected is now used")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
