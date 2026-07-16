"""
database.py
-----------
Downloads the Olist dataset from Kaggle and loads all 9 CSVs into SQLite.

Usage:
    python database.py

Requirements:
    - Kaggle credentials in ~/.kaggle/kaggle.json
    - pip install pandas kaggle
"""

import os
import sqlite3
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "raw"
DB_PATH    = BASE_DIR / "data" / "olist.db"
SCHEMA_SQL = BASE_DIR / "sql" / "schema.sql"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

CSV_TABLE_MAP = {
    "olist_customers_dataset.csv":           "customers",
    "olist_geolocation_dataset.csv":         "geolocation",
    "olist_sellers_dataset.csv":             "sellers",
    "olist_products_dataset.csv":            "products",
    "olist_orders_dataset.csv":              "orders",
    "olist_order_items_dataset.csv":         "order_items",
    "olist_order_payments_dataset.csv":      "payments",
    "olist_order_reviews_dataset.csv":       "reviews",
    "product_category_name_translation.csv": "product_category_translation",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def create_database(conn: sqlite3.Connection) -> None:
    """Apply schema.sql — creates all tables and indexes."""
    schema = SCHEMA_SQL.read_text()
    conn.executescript(schema)
    log.info("Schema applied.")


def import_tables(conn: sqlite3.Connection) -> None:
    """Read each CSV and write it into the SQLite database."""
    for filename, table in CSV_TABLE_MAP.items():
        path = DATA_DIR / filename
        if not path.exists():
            log.warning("Missing: %s — skipped.", filename)
            continue
        df = pd.read_csv(path, low_memory=False)
        df.to_sql(table, conn, if_exists="replace", index=False)
        log.info("  %-45s → %-30s  %d rows", filename, table, len(df))


def validate_database(conn: sqlite3.Connection) -> None:
    """Print row counts for all tables as a sanity check."""
    log.info("--- Validation ---")
    for table in CSV_TABLE_MAP.values():
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        log.info("  %-30s  %d rows", table, count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not any(DATA_DIR.glob("*.csv")):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Downloading Olist dataset from Kaggle ...")
        os.system(f"kaggle datasets download -d {KAGGLE_DATASET} --path {DATA_DIR} --unzip")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    try:
        create_database(conn)
        import_tables(conn)
        conn.commit()
        validate_database(conn)
        log.info("Done. Database ready at: %s", DB_PATH)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
