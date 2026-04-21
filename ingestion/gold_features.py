"""
Gold Layer — Feature Engineering for ML
Creates ML-ready datasets with advanced features for PhD-level analysis.
"""
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import SILVER_DIR, GOLD_DIR, COMPRESSION

logger = logging.getLogger("gold_features")


class GoldFeaturePipeline:
    """Engineer ML-ready features from Silver layer data."""

    def __init__(self):
        self.silver = SILVER_DIR
        self.gold = GOLD_DIR

    def build_race_prediction_features(self) -> pd.DataFrame:
        """
        Build the main ML feature table for race outcome prediction.
        Target: final position (regression) or podium/points (classification).
        """
        master = pd.read_parquet(self.silver / "race_master.parquet")

        # ── Rolling Driver Form (last 5 races) ──
        master = master.sort_values(["driver_id", "season", "round"])
        grp = master.groupby("driver_id")

        master["rolling_avg_position_5"] = grp["position"].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        master["rolling_avg_points_5"] = grp["points"].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        master["rolling_dnf_rate_5"] = grp["dnf"].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        master["rolling_positions_gained_5"] = grp["position_gained"].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )

        # ── Constructor Strength ──
        constructor_grp = master.groupby(["season", "round", "constructor_id"])
        constructor_avg = master.groupby("constructor_id")["points"].transform(
            lambda x: x.rolling(10, min_periods=1).mean().shift(1)
        )
        master["constructor_rolling_points"] = constructor_avg

        # ── Grid vs Finish Consistency ──
        master["grid_finish_delta"] = master["grid"] - master["position"].fillna(20)
        master["rolling_consistency"] = grp["grid_finish_delta"].transform(
            lambda x: x.rolling(5, min_periods=1).std().shift(1)
        )

        # ── Qualifying Pace Relative to Field ──
        race_best_quali = master.groupby(["season", "round"])["best_quali_seconds"].transform("min")
        master["quali_gap_to_pole"] = master["best_quali_seconds"] - race_best_quali
        master["quali_gap_pct"] = (master["quali_gap_to_pole"] / race_best_quali) * 100

        # ── Pit Stop Strategy Features ──
        master["pit_strategy_aggressive"] = (master["num_pit_stops"] >= 3).astype(int)
        race_avg_pits = master.groupby(["season", "round"])["num_pit_stops"].transform("mean")
        master["pit_stops_vs_field"] = master["num_pit_stops"] - race_avg_pits

        # ── Circuit Historical Performance ──
        circuit_grp = master.groupby(["driver_id", "circuit_id"])
        master["circuit_avg_position"] = circuit_grp["position"].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        master["circuit_races_count"] = circuit_grp["position"].transform(
            lambda x: x.expanding().count().shift(1)
        )

        # ── Season Momentum ──
        season_grp = master.groupby(["driver_id", "season"])
        master["season_cumulative_points"] = season_grp["points"].cumsum() - master["points"]

        # ── Target Variables ──
        master["target_position"] = master["position"]
        master["target_podium"] = (master["position"] <= 3).astype(int)
        master["target_points_finish"] = (master["position"] <= 10).astype(int)
        master["target_winner"] = (master["position"] == 1).astype(int)

        # ── Drop Leaky Columns ──
        leaky_cols = [
            "points", "status", "laps_completed", "time_millis",
            "fastest_lap_rank", "fastest_lap_time", "fastest_lap_seconds",
            "avg_speed_kph", "finished", "dnf", "position_gained",
            "grid_finish_delta", "had_slow_stop", "total_pit_time",
            "avg_pit_duration",
        ]
        feature_df = master.drop(columns=[c for c in leaky_cols if c in master.columns],
                                 errors="ignore")

        # ── Save ──
        path = self.gold / "race_prediction_features.parquet"
        feature_df.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Gold race_prediction_features: {len(feature_df)} rows, "
                    f"{len(feature_df.columns)} columns")

        # Feature catalog
        catalog = {
            "name": "race_prediction_features",
            "created_at": datetime.utcnow().isoformat(),
            "rows": len(feature_df),
            "features": list(feature_df.columns),
            "targets": ["target_position", "target_podium", "target_points_finish", "target_winner"],
            "feature_descriptions": {
                "rolling_avg_position_5": "Average finish position over last 5 races",
                "rolling_avg_points_5": "Average points scored over last 5 races",
                "rolling_dnf_rate_5": "DNF rate over last 5 races",
                "constructor_rolling_points": "Constructor average points (rolling 10 races)",
                "quali_gap_to_pole": "Qualifying time gap to pole position (seconds)",
                "quali_gap_pct": "Qualifying gap as percentage of pole time",
                "circuit_avg_position": "Driver historical avg position at this circuit",
                "circuit_races_count": "Number of previous races at this circuit",
                "season_cumulative_points": "Cumulative points in the current season so far",
                "rolling_consistency": "Std dev of grid-to-finish delta (lower = more consistent)",
                "pit_stops_vs_field": "Number of pit stops relative to field average",
            }
        }
        import json
        with open(self.gold / "feature_catalog.json", "w") as f:
            json.dump(catalog, f, indent=2)

        return feature_df

    def build_tire_degradation_features(self) -> pd.DataFrame:
        """
        Build tire degradation curves from lap data — PhD-level feature.
        Models how lap time degrades as tire life increases per compound.
        """
        laps_path = self.silver / "laps.parquet"
        if not laps_path.exists():
            logger.warning("No laps data in Silver layer — skipping tire features")
            return pd.DataFrame()

        laps = pd.read_parquet(laps_path)

        if "LapTime_seconds" not in laps.columns or "TyreLife" not in laps.columns:
            logger.warning("Missing LapTime or TyreLife columns")
            return pd.DataFrame()

        # Filter out pit laps, safety car laps, outliers
        laps = laps[laps["LapTime_seconds"].notna()]
        laps = laps[laps["LapTime_seconds"] > 0]

        # Remove extreme outliers (pit in/out laps, safety car)
        race_median = laps.groupby(["season", "round"])["LapTime_seconds"].transform("median")
        laps = laps[laps["LapTime_seconds"] < race_median * 1.15]

        # Per-stint degradation slope
       
        
        def stint_degradation(group):
            if len(group) < 3:
                return pd.Series({"tire_deg_slope": np.nan, "tire_deg_r2": np.nan})
            x = group["TyreLife"].values.astype(float)
            y = group["LapTime_seconds"].values.astype(float)
            # Remove NaN/Inf values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 3 or np.std(x) == 0:
                return pd.Series({"tire_deg_slope": np.nan, "tire_deg_r2": np.nan})
            try:
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                return pd.Series({"tire_deg_slope": coeffs[0], "tire_deg_r2": r2})
            except Exception:
                return pd.Series({"tire_deg_slope": np.nan, "tire_deg_r2": np.nan})

        if "Stint" in laps.columns:
            stint_key = ["season", "round", "Driver", "Stint"]
        else:
            stint_key = ["season", "round", "Driver", "Compound"]

        deg_features = laps.groupby(stint_key).apply(stint_degradation).reset_index()

        path = self.gold / "tire_degradation_features.parquet"
        deg_features.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Gold tire_degradation_features: {len(deg_features)} rows")
        return deg_features

    def run_all(self):
        """Build all Gold feature tables."""
        logger.info("=" * 60)
        logger.info("GOLD LAYER FEATURE ENGINEERING")
        logger.info("=" * 60)

        self.build_race_prediction_features()
        self.build_tire_degradation_features()
        logger.info("Gold layer complete ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    GoldFeaturePipeline().run_all()
