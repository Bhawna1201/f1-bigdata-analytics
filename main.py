"""
F1 Big Data Pipeline — Main Orchestrator
Run the full Bronze → Silver → Gold pipeline.

Usage:
    python main.py                        # Full pipeline, all seasons, no telemetry
    python main.py --season 2023          # Single season
    python main.py --telemetry            # Include car telemetry (slow/large)
    python main.py --layer silver         # Run only silver+ layers (assumes bronze exists)
    python main.py --layer gold           # Run only gold layer (assumes silver exists)
"""
import argparse
import logging
import time
from datetime import datetime

from ingestion.bronze_ingestion import BronzeIngestionPipeline
from ingestion.silver_transform import SilverTransformPipeline
from ingestion.gold_features import GoldFeaturePipeline
from config.settings import LOG_FORMAT, SEASONS_START, SEASONS_END


def main():
    parser = argparse.ArgumentParser(description="F1 Big Data — Full Pipeline")
    parser.add_argument("--season", type=int, help="Single season")
    parser.add_argument("--start", type=int, default=SEASONS_START)
    parser.add_argument("--end", type=int, default=SEASONS_END)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--layer", choices=["bronze", "silver", "gold"],
                        default="bronze", help="Start from this layer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("main")

    start_time = time.time()
    logger.info(f"F1 Big Data Pipeline started at {datetime.utcnow().isoformat()}")

    # Bronze
    if args.layer == "bronze":
        bronze = BronzeIngestionPipeline()
        if args.season:
            bronze.ingest_season(args.season, include_telemetry=args.telemetry)
        else:
            bronze.ingest_all_seasons(args.start, args.end,
                                      include_telemetry=args.telemetry)

    # Silver
    if args.layer in ("bronze", "silver"):
        SilverTransformPipeline().run_all()

    # Gold
    GoldFeaturePipeline().run_all()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
