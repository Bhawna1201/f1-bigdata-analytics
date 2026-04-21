"""Run full ingestion with smart rate-limit handling for ALL APIs."""
import time
import logging
import fastf1
from ingestion.bronze_ingestion import ErgastIngestion, FastF1Ingestion
from ingestion.silver_transform import SilverTransformPipeline
from ingestion.gold_features import GoldFeaturePipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_all")


def retry_with_backoff(func, *args, max_retries=5, base_wait=60, **kwargs):
    """Retry any function with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many" in err or "RateLimit" in err or "500 calls" in err:
                wait = base_wait * (attempt + 1)
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}) — waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Non-rate-limit error: {e}")
                return None
    logger.error("Max retries exceeded — skipping")
    return None


# ── Step 1: Pull ALL Ergast data with retry ──
ergast = ErgastIngestion()
for season in range(2018, 2025):
    logger.info(f"{'='*40} Ergast → {season} {'='*40}")
    retry_with_backoff(ergast.ingest_race_results, season)
    time.sleep(2)
    retry_with_backoff(ergast.ingest_qualifying, season)
    time.sleep(2)
    retry_with_backoff(ergast.ingest_pit_stops, season)
    time.sleep(2)
    retry_with_backoff(ergast.ingest_driver_standings, season)
    time.sleep(5)  # Extra pause between seasons

logger.info("✅ Ergast complete for all seasons")

# ── Step 2: Pull FastF1 with auto-retry on rate limit ──
ff1 = FastF1Ingestion()

for season in range(2018, 2025):
    logger.info(f"{'='*40} FastF1 → {season} {'='*40}")

    schedule = retry_with_backoff(fastf1.get_event_schedule, season)
    if schedule is None:
        logger.error(f"Could not get schedule for {season} — skipping")
        continue

    rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()

    for rnd in rounds:
        if rnd == 0:
            continue
        logger.info(f"  FastF1 → {season} R{rnd}")
        retry_with_backoff(ff1.ingest_session_laps, season, rnd, "R",
                           base_wait=120)
        retry_with_backoff(ff1.ingest_weather, season, rnd, "R",
                           base_wait=120)
        time.sleep(5)

logger.info("✅ FastF1 complete")

# ── Step 3: Silver + Gold ──
logger.info("Running Silver transforms...")
SilverTransformPipeline().run_all()
logger.info("Running Gold features...")
GoldFeaturePipeline().run_all()
logger.info("🏁 ALL DONE")
