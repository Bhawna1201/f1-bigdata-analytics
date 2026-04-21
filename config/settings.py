"""
F1 Big Data Project — Configuration
All free-tier compatible settings
"""
import os
from pathlib import Path

# === Project Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Data Source Settings ===
ERGAST_BASE_URL = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE_URL = "https://api.openf1.org/v1"
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# === Season Range ===
SEASONS_START = 2018  # FastF1 telemetry available from 2018+
SEASONS_END = 2024

# === Storage Format ===
STORAGE_FORMAT = "parquet"  # parquet for lakehouse compatibility
COMPRESSION = "snappy"

# === Medallion Architecture ===
# Bronze: Raw data as-is from APIs
# Silver: Cleaned, typed, deduplicated, joined
# Gold:   Aggregated, feature-engineered, ML-ready

MEDALLION_LAYERS = {
    "bronze": {"path": BRONZE_DIR, "description": "Raw ingested data"},
    "silver": {"path": SILVER_DIR, "description": "Cleaned and validated"},
    "gold":   {"path": GOLD_DIR,   "description": "ML-ready features"},
}

# === Logging ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
