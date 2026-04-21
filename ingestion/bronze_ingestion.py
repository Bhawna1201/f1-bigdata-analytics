"""
Bronze Layer Ingestion — Raw Data from F1 APIs
Pulls data from FastF1, Ergast, and OpenF1 into Parquet files.
No transformations — raw data preservation for reproducibility.
"""
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    BRONZE_DIR, ERGAST_BASE_URL, OPENF1_BASE_URL,
    SEASONS_START, SEASONS_END, COMPRESSION
)

logger = logging.getLogger("bronze_ingestion")


# ─── Ergast API (Race Results, Standings, Pit Stops) ──────────────────────

class ErgastIngestion:
    """Ingest historical race data from Ergast API."""

    def __init__(self):
        self.base_url = ERGAST_BASE_URL
        self.output_dir = BRONZE_DIR / "ergast"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(self, endpoint: str, limit: int = 100) -> list:
        """Fetch with pagination and retry on 429."""
        all_data = None
        offset = 0
        while True:
            url = f"{self.base_url}/{endpoint}.json?limit={limit}&offset={offset}"
            logger.info(f"Fetching: {url}")

            for attempt in range(5):
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"429 rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                logger.error(f"Max retries for {url}")
                return all_data or {"MRData": {"total": "0"}}

            data = resp.json()
            total = int(data["MRData"]["total"])

            if all_data is None:
                all_data = data
            else:
                table_key = [k for k in data["MRData"] if k.endswith("Table")][0]
                list_key = [k for k in data["MRData"][table_key] if isinstance(data["MRData"][table_key][k], list)][0]
                all_data["MRData"][table_key][list_key].extend(
                    data["MRData"][table_key][list_key]
                )

            offset += limit
            if offset >= total:
                break
            time.sleep(1)

        return all_data
    

    def ingest_race_results(self, season: int) -> pd.DataFrame:
        """Pull all race results for a season."""
        data = self._fetch(f"{season}/results")
        races = data["MRData"]["RaceTable"]["Races"]

        rows = []
        for race in races:
            for result in race.get("Results", []):
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "circuit_name": race["Circuit"]["circuitName"],
                    "race_date": race["date"],
                    "driver_id": result["Driver"]["driverId"],
                    "driver_code": result["Driver"].get("code", ""),
                    "driver_name": f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                    "constructor_id": result["Constructor"]["constructorId"],
                    "constructor_name": result["Constructor"]["name"],
                    "grid": int(result["grid"]),
                    "position": int(result["position"]) if result["position"].isdigit() else None,
                    "points": float(result["points"]),
                    "status": result["status"],
                    "laps_completed": int(result["laps"]),
                    "time_millis": int(result["Time"]["millis"]) if "Time" in result else None,
                    "fastest_lap_rank": result.get("FastestLap", {}).get("rank"),
                    "fastest_lap_time": result.get("FastestLap", {}).get("Time", {}).get("time"),
                    "avg_speed_kph": result.get("FastestLap", {}).get("AverageSpeed", {}).get("speed"),
                    "_ingested_at": datetime.utcnow().isoformat(),
                })

        df = pd.DataFrame(rows)
        path = self.output_dir / f"race_results_{season}.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Ergast race results {season}: {len(df)} rows → {path}")
        return df

    def ingest_qualifying(self, season: int) -> pd.DataFrame:
        """Pull qualifying results for a season."""
        data = self._fetch(f"{season}/qualifying")
        races = data["MRData"]["RaceTable"]["Races"]

        rows = []
        for race in races:
            for q in race.get("QualifyingResults", []):
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "driver_id": q["Driver"]["driverId"],
                    "driver_name": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
                    "constructor_id": q["Constructor"]["constructorId"],
                    "position": int(q["position"]),
                    "q1": q.get("Q1"),
                    "q2": q.get("Q2"),
                    "q3": q.get("Q3"),
                    "_ingested_at": datetime.utcnow().isoformat(),
                })

        df = pd.DataFrame(rows)
        path = self.output_dir / f"qualifying_{season}.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Ergast qualifying {season}: {len(df)} rows → {path}")
        return df

    def ingest_pit_stops(self, season: int) -> pd.DataFrame:
        """Pull pit stop data for all races in a season."""
        # First get the race schedule to know how many rounds
        schedule = self._fetch(f"{season}")
        races = schedule["MRData"]["RaceTable"]["Races"]

        all_rows = []
        for race in races:
            rnd = race["round"]
            try:
                data = self._fetch(f"{season}/{rnd}/pitstops")
                stops = data["MRData"]["RaceTable"]["Races"]
                if stops:
                    for stop in stops[0].get("PitStops", []):
                        all_rows.append({
                            "season": int(season),
                            "round": int(rnd),
                            "race_name": race["raceName"],
                            "driver_id": stop["driverId"],
                            "stop_number": int(stop["stop"]),
                            "lap": int(stop["lap"]),
                            "time_of_day": stop["time"],
                            "duration": stop["duration"],
                            "_ingested_at": datetime.utcnow().isoformat(),
                        })
                time.sleep(0.2)  # Rate limit respect
            except Exception as e:
                logger.warning(f"Pit stops {season} R{rnd}: {e}")

        df = pd.DataFrame(all_rows)
        path = self.output_dir / f"pit_stops_{season}.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Ergast pit stops {season}: {len(df)} rows → {path}")
        return df

    def ingest_driver_standings(self, season: int) -> pd.DataFrame:
        """Pull final driver standings for a season."""
        data = self._fetch(f"{season}/driverStandings")
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"]

        rows = []
        if standings_list:
            for s in standings_list[0]["DriverStandings"]:
                rows.append({
                    "season": season,
                    "position": int(s["position"]),
                    "points": float(s["points"]),
                    "wins": int(s["wins"]),
                    "driver_id": s["Driver"]["driverId"],
                    "driver_name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                    "constructor_id": s["Constructors"][0]["constructorId"],
                    "constructor_name": s["Constructors"][0]["name"],
                    "_ingested_at": datetime.utcnow().isoformat(),
                })

        df = pd.DataFrame(rows)
        path = self.output_dir / f"driver_standings_{season}.parquet"
        df.to_parquet(path, compression=COMPRESSION, index=False)
        logger.info(f"✓ Ergast standings {season}: {len(df)} rows → {path}")
        return df


# ─── FastF1 (Telemetry: Speed, Throttle, Brake, Gear, DRS) ─────────────

class FastF1Ingestion:
    """Ingest telemetry and lap data via FastF1 library."""

    def __init__(self):
        self.output_dir = BRONZE_DIR / "fastf1"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = BRONZE_DIR / "fastf1_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

    def ingest_session_laps(self, season: int, round_num: int,
                            session_type: str = "R") -> Optional[pd.DataFrame]:
        """
        Pull lap-by-lap data for a session.
        session_type: 'R' = Race, 'Q' = Qualifying, 'FP1/FP2/FP3'
        """
        try:
            session = fastf1.get_session(season, round_num, session_type)
            session.load(telemetry=False, weather=True)

            laps = session.laps.copy()
            if laps.empty:
                logger.warning(f"No lap data: {season} R{round_num} {session_type}")
                return None

            # Add metadata
            laps["season"] = season
            laps["round"] = round_num
            laps["session_type"] = session_type
            laps["event_name"] = session.event["EventName"]
            laps["_ingested_at"] = datetime.utcnow().isoformat()

            # Convert timedelta columns to seconds for parquet compatibility
            td_cols = laps.select_dtypes(include=["timedelta64"]).columns
            for col in td_cols:
                laps[f"{col}_seconds"] = laps[col].dt.total_seconds()
                laps = laps.drop(columns=[col])

            path = self.output_dir / f"laps_{season}_R{round_num}_{session_type}.parquet"
            laps.to_parquet(path, compression=COMPRESSION, index=False)
            logger.info(f"✓ FastF1 laps {season} R{round_num} {session_type}: {len(laps)} rows")
            return laps

        except Exception as e:
            logger.error(f"FastF1 laps {season} R{round_num}: {e}")
            return None

    def ingest_telemetry(self, season: int, round_num: int,
                         session_type: str = "R",
                         drivers: Optional[list] = None) -> Optional[pd.DataFrame]:
        """
        Pull car telemetry (speed, throttle, brake, gear, DRS, RPM).
        This is the high-frequency data — thousands of rows per lap.
        """
        try:
            session = fastf1.get_session(season, round_num, session_type)
            session.load(telemetry=True, weather=False)

            all_tel = []
            driver_list = drivers or session.laps["Driver"].unique().tolist()

            for driver in driver_list:
                driver_laps = session.laps.pick_driver(driver)
                for _, lap in driver_laps.iterrows():
                    try:
                        tel = lap.get_telemetry()
                        if tel is not None and not tel.empty:
                            tel = tel.copy()
                            tel["Driver"] = driver
                            tel["LapNumber"] = lap["LapNumber"]
                            tel["Compound"] = lap.get("Compound", "UNKNOWN")
                            tel["TyreLife"] = lap.get("TyreLife", None)
                            all_tel.append(tel)
                    except Exception:
                        continue

            if not all_tel:
                return None

            df = pd.concat(all_tel, ignore_index=True)
            df["season"] = season
            df["round"] = round_num
            df["session_type"] = session_type
            df["_ingested_at"] = datetime.utcnow().isoformat()

            # Handle timedelta columns
            td_cols = df.select_dtypes(include=["timedelta64"]).columns
            for col in td_cols:
                df[f"{col}_seconds"] = df[col].dt.total_seconds()
                df = df.drop(columns=[col])

            path = self.output_dir / f"telemetry_{season}_R{round_num}_{session_type}.parquet"
            df.to_parquet(path, compression=COMPRESSION, index=False)
            logger.info(f"✓ FastF1 telemetry {season} R{round_num}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"FastF1 telemetry {season} R{round_num}: {e}")
            return None

    def ingest_weather(self, season: int, round_num: int,
                       session_type: str = "R") -> Optional[pd.DataFrame]:
        """Pull session weather data."""
        try:
            session = fastf1.get_session(season, round_num, session_type)
            session.load(telemetry=False, weather=True)

            weather = session.weather_data.copy()
            if weather.empty:
                return None

            weather["season"] = season
            weather["round"] = round_num
            weather["session_type"] = session_type
            weather["_ingested_at"] = datetime.utcnow().isoformat()

            # Handle timedelta
            td_cols = weather.select_dtypes(include=["timedelta64"]).columns
            for col in td_cols:
                weather[f"{col}_seconds"] = weather[col].dt.total_seconds()
                weather = weather.drop(columns=[col])

            path = self.output_dir / f"weather_{season}_R{round_num}_{session_type}.parquet"
            weather.to_parquet(path, compression=COMPRESSION, index=False)
            logger.info(f"✓ FastF1 weather {season} R{round_num}: {len(weather)} rows")
            return weather

        except Exception as e:
            logger.error(f"FastF1 weather {season} R{round_num}: {e}")
            return None


# ─── Full Season Ingestion Orchestrator ──────────────────────────────────

class BronzeIngestionPipeline:
    """Orchestrate full bronze-layer ingestion for one or more seasons."""

    def __init__(self):
        self.ergast = ErgastIngestion()
        self.fastf1 = FastF1Ingestion()
        self.manifest = []

    def ingest_season(self, season: int, include_telemetry: bool = False):
        """
        Full ingestion for a season.
        Set include_telemetry=True for detailed car data (slow, large).
        """
        logger.info(f"{'='*60}")
        logger.info(f"INGESTING SEASON {season}")
        logger.info(f"{'='*60}")

        # 1. Ergast: structured race data
        self.ergast.ingest_race_results(season)
        self.ergast.ingest_qualifying(season)
        self.ergast.ingest_pit_stops(season)
        self.ergast.ingest_driver_standings(season)

        # 2. FastF1: lap data + weather for each round
        schedule = fastf1.get_event_schedule(season)
        race_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()

        for rnd in race_rounds:
            if rnd == 0:
                continue
            logger.info(f"  → Season {season}, Round {rnd}")
            try:
                self.fastf1.ingest_session_laps(season, rnd, "R")
                self.fastf1.ingest_weather(season, rnd, "R")
            except Exception as e:
                if "500 calls" in str(e) or "RateLimit" in str(e):
                    logger.warning(f"Rate limited — waiting 10 minutes then retrying...")
                    time.sleep(600)
                    self.fastf1.ingest_session_laps(season, rnd, "R")
                    self.fastf1.ingest_weather(season, rnd, "R")
                else:
                    logger.error(f"FastF1 error: {e}")

            if include_telemetry:
                self.fastf1.ingest_telemetry(season, rnd, "R")

            time.sleep(5)  # Be nice to APIs

        self.manifest.append({
            "season": season,
            "ingested_at": datetime.utcnow().isoformat(),
            "telemetry": include_telemetry,
        })

    def ingest_all_seasons(self, start: int = SEASONS_START,
                           end: int = SEASONS_END,
                           include_telemetry: bool = False):
        """Ingest all configured seasons."""
        for season in range(start, end + 1):
            self.ingest_season(season, include_telemetry=include_telemetry)

        # Save manifest
        manifest_path = BRONZE_DIR / "ingestion_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        logger.info(f"Manifest saved: {manifest_path}")


# ─── CLI Entry Point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="F1 Bronze Layer Ingestion")
    parser.add_argument("--season", type=int, help="Single season to ingest")
    parser.add_argument("--start", type=int, default=SEASONS_START)
    parser.add_argument("--end", type=int, default=SEASONS_END)
    parser.add_argument("--telemetry", action="store_true",
                        help="Include high-frequency car telemetry")
    args = parser.parse_args()

    pipeline = BronzeIngestionPipeline()

    if args.season:
        pipeline.ingest_season(args.season, include_telemetry=args.telemetry)
    else:
        pipeline.ingest_all_seasons(args.start, args.end,
                                    include_telemetry=args.telemetry)
