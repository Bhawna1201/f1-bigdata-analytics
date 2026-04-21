"""
Silver Layer — Clean, Validate, Deduplicate, Join
Transforms raw Bronze data into analysis-ready datasets.
"""
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import BRONZE_DIR, SILVER_DIR, COMPRESSION

logger = logging.getLogger("silver_transform")


class SilverTransformPipeline:
    """Transform Bronze → Silver with data quality checks."""

    def __init__(self):
        self.bronze = BRONZE_DIR
        self.silver = SILVER_DIR
        self.quality_report = {}

    # ─── Helpers ──────────────────────────────────────────────────

    def _read_bronze_parquets(self, subdir: str, pattern: str) -> pd.DataFrame:
        """Read and concatenate all matching parquet files."""
        path = self.bronze / subdir
        files = sorted(path.glob(pattern))
        if not files:
            logger.warning(f"No files matching {subdir}/{pattern}")
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Read {len(files)} files from {subdir}/{pattern}: {len(combined)} total rows")
        return combined

    def _log_quality(self, name: str, df: pd.DataFrame, nulls_before: int, nulls_after: int,
                     dupes_removed: int):
        self.quality_report[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "nulls_before": nulls_before,
            "nulls_after": nulls_after,
            "duplicates_removed": dupes_removed,
            "processed_at": datetime.utcnow().isoformat(),
        }

    # ─── Race Results ─────────────────────────────────────────────

    def transform_race_results(self) -> pd.DataFrame:
        """Clean race results: type casting, dedup, null handling."""
        df = self._read_bronze_parquets("ergast", "race_results_*.parquet")
        if df.empty:
            return df

        nulls_before = int(df.isnull().sum().sum())
        len_before = len(df)

        # Deduplicate
        df = df.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
        dupes = len_before - len(df)

        # Type enforcement
        df["race_date"] = pd.to_datetime(df["race_date"])
        df["points"] = df["points"].astype(float)
        df["grid"] = df["grid"].astype(int)

        # Parse fastest lap time to seconds
        def parse_lap_time(t):
            if pd.isna(t) or not isinstance(t, str):
                return np.nan
            parts = t.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return np.nan

        df["fastest_lap_seconds"] = df["fastest_lap_time"].apply(parse_lap_time)
        df["avg_speed_kph"] = pd.to_numeric(df["avg_speed_kph"], errors="coerce")

        # Derived columns
        df["position_gained"] = df["grid"] - df["position"].fillna(df["grid"])
        df["finished"] = df["status"].str.lower().str.contains("finished|\\+", na=False)
        df["dnf"] = ~df["finished"]

        nulls_after = int(df.isnull().sum().sum())

        # Drop ingestion metadata
        df = df.drop(columns=["_ingested_at"], errors="ignore")

        path = self.silver / "race_results.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        self._log_quality("race_results", df, nulls_before, nulls_after, dupes)
        logger.info(f"✓ Silver race_results: {len(df)} rows, {dupes} dupes removed")
        return df

    # ─── Qualifying ───────────────────────────────────────────────

    def transform_qualifying(self) -> pd.DataFrame:
        """Clean qualifying: parse Q1/Q2/Q3 times to seconds."""
        df = self._read_bronze_parquets("ergast", "qualifying_*.parquet")
        if df.empty:
            return df

        nulls_before = int(df.isnull().sum().sum())
        len_before = len(df)
        df = df.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
        dupes = len_before - len(df)

        def parse_quali_time(t):
            if pd.isna(t) or not isinstance(t, str):
                return np.nan
            parts = t.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return np.nan

        for col in ["q1", "q2", "q3"]:
            df[f"{col}_seconds"] = df[col].apply(parse_quali_time)

        # Best qualifying time
        df["best_quali_seconds"] = df[["q1_seconds", "q2_seconds", "q3_seconds"]].min(axis=1)

        df = df.drop(columns=["_ingested_at"], errors="ignore")
        nulls_after = int(df.isnull().sum().sum())

        path = self.silver / "qualifying.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        self._log_quality("qualifying", df, nulls_before, nulls_after, dupes)
        logger.info(f"✓ Silver qualifying: {len(df)} rows")
        return df

    # ─── Pit Stops ────────────────────────────────────────────────

    def transform_pit_stops(self) -> pd.DataFrame:
        """Clean pit stops: parse duration to float seconds."""
        df = self._read_bronze_parquets("ergast", "pit_stops_*.parquet")
        if df.empty:
            return df

        nulls_before = int(df.isnull().sum().sum())
        len_before = len(df)
        df = df.drop_duplicates(
            subset=["season", "round", "driver_id", "stop_number"], keep="last"
        )
        dupes = len_before - len(df)

        # Parse duration string to seconds
        df["duration_seconds"] = pd.to_numeric(df["duration"], errors="coerce")

        # Flag slow stops (> 5s is a slow stop, > 15s likely issue/penalty)
        df["is_slow_stop"] = df["duration_seconds"] > 5.0
        df["is_problem_stop"] = df["duration_seconds"] > 15.0

        df = df.drop(columns=["_ingested_at"], errors="ignore")
        nulls_after = int(df.isnull().sum().sum())

        path = self.silver / "pit_stops.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        self._log_quality("pit_stops", df, nulls_before, nulls_after, dupes)
        logger.info(f"✓ Silver pit_stops: {len(df)} rows")
        return df

    # ─── Lap Data ─────────────────────────────────────────────────

    def transform_laps(self) -> pd.DataFrame:
        """Clean FastF1 lap data: normalize times, add stint info."""
        df = self._read_bronze_parquets("fastf1", "laps_*.parquet")
        if df.empty:
            return df

        nulls_before = int(df.isnull().sum().sum())
        len_before = len(df)
        df = df.drop_duplicates(
            subset=["season", "round", "Driver", "LapNumber"], keep="last"
        )
        dupes = len_before - len(df)

        # Normalize column names
        df.columns = [c.replace(" ", "_") for c in df.columns]

        # Calculate lap delta from leader (if LapTime available)
        if "LapTime_seconds" in df.columns:
            leader_times = df.groupby(["season", "round", "LapNumber"])["LapTime_seconds"].transform("min")
            df["delta_to_leader_seconds"] = df["LapTime_seconds"] - leader_times

        # Compound encoding for ML
        compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
        if "Compound" in df.columns:
            df["compound_encoded"] = df["Compound"].map(compound_map).fillna(-1).astype(int)

        df = df.drop(columns=["_ingested_at"], errors="ignore")
        nulls_after = int(df.isnull().sum().sum())

        path = self.silver / "laps.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        self._log_quality("laps", df, nulls_before, nulls_after, dupes)
        logger.info(f"✓ Silver laps: {len(df)} rows")
        return df

    # ─── Joined Race Dataset ──────────────────────────────────────

    def build_race_master(self) -> pd.DataFrame:
        """
        Join race results + qualifying + pit stop aggregates → single race-driver dataset.
        This is the core Silver table for downstream ML.
        """
        results = pd.read_parquet(self.silver / "race_results.parquet")
        quali = pd.read_parquet(self.silver / "qualifying.parquet")
        pits = pd.read_parquet(self.silver / "pit_stops.parquet")

        # Aggregate pit stops per driver per race
        pit_agg = pits.groupby(["season", "round", "driver_id"]).agg(
            num_pit_stops=("stop_number", "max"),
            avg_pit_duration=("duration_seconds", "mean"),
            total_pit_time=("duration_seconds", "sum"),
            had_slow_stop=("is_slow_stop", "any"),
        ).reset_index()

        # Join qualifying
        master = results.merge(
            quali[["season", "round", "driver_id", "best_quali_seconds",
                   "q1_seconds", "q2_seconds", "q3_seconds"]],
            on=["season", "round", "driver_id"],
            how="left"
        )

        # Join pit stops
        master = master.merge(pit_agg, on=["season", "round", "driver_id"], how="left")

        # Fill missing pit data (e.g., DNF before pit)
        master["num_pit_stops"] = master["num_pit_stops"].fillna(0).astype(int)

        path = self.silver / "race_master.parquet"
        master.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Silver race_master: {len(master)} rows, {len(master.columns)} columns")
        return master

    # ─── Run All ──────────────────────────────────────────────────

    def run_all(self):
        """Execute full Silver transformation pipeline."""
        logger.info("=" * 60)
        logger.info("SILVER LAYER TRANSFORMATION")
        logger.info("=" * 60)

        self.transform_race_results()
        self.transform_qualifying()
        self.transform_pit_stops()
        self.transform_laps()
        self.build_race_master()

        # Save quality report
        import json
        report_path = self.silver / "quality_report.json"
        with open(report_path, "w") as f:
            json.dump(self.quality_report, f, indent=2)
        logger.info(f"Quality report: {report_path}")
        logger.info("Silver layer complete ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    SilverTransformPipeline().run_all()
