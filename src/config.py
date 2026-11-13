"""Central configuration: paths and constants used across the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "warehouse.db"
SQL_DIR = ROOT / "sql"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_CUSTOMERS = 5000
