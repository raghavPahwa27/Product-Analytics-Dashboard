"""
database.py
-----------
Downloads the Olist dataset from Kaggle and loads all 9 CSVs into SQLite.

Usage:
    python database.py

Requirements:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
      (or KAGGLE_USERNAME / KAGGLE_KEY env vars in a .env file)
    - pip install -r requirements.txt

Design decisions:
    - SQLAlchemy engine abstracts the SQLite connection; easy to swap to
      Postgres later by changing the connection string.
    - pandas `to_sql` with `if_exists="replace"` makes the script idempotent —
      safe to re-run without manual cleanup.
    - Chunked writes (chunksize=500) avoid loading huge tables into memory all
      at once.
    - Schema is applied from schema.sql before any data lands, so FK
      relationships and indexes are always consistent.
"""

import os
import zipfile
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "raw"
DB_PATH    = BASE_DIR / "data" / "olist.db"
SCHEMA_SQL = BASE_DIR / "sql" / "schema.sql"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

# Maps Kaggle CSV filename → SQLite table name
CSV_TABLE_MAP = {
    "olist_customers_dataset.csv":             "customers",
    "olist_geolocation_dataset.csv":           "geolocation",
    "olist_sellers_dataset.csv":               "sellers",
    "olist_products_dataset.csv":              "products",
    "olist_orders_dataset.csv":                "orders",
    "olist_order_items_dataset.csv":           "order_items",
    "olist_order_payments_dataset.csv":        "payments",
    "olist_order_reviews_dataset.csv":         "reviews",
    "product_category_name_translation.csv":   "product_category_translation",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_dataset() -> None:
    """Pull the Olist zip from Kaggle if the raw folder is empty."""
    if any(DATA_DIR.glob("*.csv")):
        log.info("Raw CSVs already present — skipping download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading Olist dataset from Kaggle …")

    # kaggle CLI writes to cwd; redirect output explicitly
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} "
        f"--path {DATA_DIR} --unzip"
    )
    log.info("Download complete.")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def apply_schema(engine) -> None:
    """Execute schema.sql to create tables and indexes."""
    sql = SCHEMA_SQL.read_text()
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    log.info("Schema applied.")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
def load_csv(path: Path, table: str, engine) -> int:
    """Read one CSV and write it to SQLite. Returns row count."""
    df = pd.read_csv(path, low_memory=False)
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=500)
    return len(df)


def load_all_csvs(engine) -> None:
    """Iterate CSV_TABLE_MAP and load every file."""
    for filename, table in CSV_TABLE_MAP.items():
        path = DATA_DIR / filename
        if not path.exists():
            log.warning("Missing: %s — skipped.", filename)
            continue
        rows = load_csv(path, table, engine)
        log.info("  %-45s → %-35s  %d rows", filename, table, rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(engine) -> None:
    """Print row counts for every table as a quick sanity check."""
    tables = list(CSV_TABLE_MAP.values())
    log.info("--- Row count validation ---")
    with engine.connect() as conn:
        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            log.info("  %-35s  %d rows", table, count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    download_dataset()

    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    log.info("Database: %s", DB_PATH)

    apply_schema(engine)
    log.info("Loading CSVs …")
    load_all_csvs(engine)
    validate(engine)

    log.info("Done. Database ready at: %s", DB_PATH)


if __name__ == "__main__":
    main()
