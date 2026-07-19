"""
preprocessing.py
----------------
Loads the Olist dataset from SQLite and returns a clean, flat DataFrame
at the order-item level. This is the single source of truth consumed by
feature_engineering.py.

Run standalone to verify the output shape:
    python preprocessing.py
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/olist.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL — one query pulls everything needed; Python handles only type conversion.
# payment_sequential = 1 selects only the primary payment method per order.
#   Some orders are split across multiple payment methods (e.g. credit card
#   + voucher). Keeping only sequential = 1 gives us one row per order and
#   represents the customer's main payment choice without double-counting.
# review_score is averaged in case of duplicate reviews on the same order.
# ---------------------------------------------------------------------------
BASE_QUERY = """
SELECT
    c.customer_unique_id,
    c.customer_state,
    c.customer_city,
    o.order_id,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    COALESCE(t.product_category_name_english,
             pr.product_category_name,
             'unknown')                                  AS category,
    pay.payment_type,
    r.review_score
FROM orders o
INNER JOIN customers c
        ON o.customer_id = c.customer_id
INNER JOIN order_items oi
        ON o.order_id = oi.order_id
LEFT JOIN products pr
        ON oi.product_id = pr.product_id
LEFT JOIN product_category_translation t
        ON pr.product_category_name = t.product_category_name
LEFT JOIN (
    SELECT order_id, payment_type
    FROM   payments
    WHERE  payment_sequential = 1
) pay ON o.order_id = pay.order_id
LEFT JOIN (
    SELECT order_id, AVG(review_score) AS review_score
    FROM   reviews
    GROUP  BY order_id
) r ON o.order_id = r.order_id
WHERE o.order_status              = 'delivered'
  AND o.order_purchase_timestamp  IS NOT NULL
"""


def load_raw_data() -> pd.DataFrame:
    """Execute the base SQL query and return a raw DataFrame."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run database.py first.")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(BASE_QUERY, conn)
    conn.close()
    log.info("Loaded %d rows from SQLite.", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning rules. Each step is documented with its rationale.
    Returns a clean DataFrame — same shape, corrected types and values.
    """
    original_len = len(df)

    # 1. Parse timestamps — stored as TEXT in SQLite.
    #    errors="coerce" converts unparseable values to NaT instead of crashing.
    for col in ["order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. Drop rows where the purchase timestamp is NaT.
    #    These rows cannot contribute any time-based feature; they must go.
    df = df.dropna(subset=["order_purchase_timestamp"])

    # 3. Drop exact duplicates (same order_id + product_id + seller_id).
    #    The SQL JOIN can produce duplicates if the reviews table has
    #    multiple rows per order_id despite the GROUP BY defensive guard.
    df = df.drop_duplicates(subset=["order_id", "product_id", "seller_id"])

    # 4. Remove items with non-positive price — clearly invalid records.
    df = df[df["price"] > 0]

    # 5. Clip freight_value at 0. Negative freight is a data entry error.
    df["freight_value"] = df["freight_value"].clip(lower=0)

    # 6. Fill missing category with 'unknown'.
    #    COALESCE in SQL handles most cases; this is a Python-side safety net.
    df["category"] = df["category"].fillna("unknown")

    # 7. Fill missing payment_type with 'unknown'.
    #    A small number of orders may have no payment record.
    df["payment_type"] = df["payment_type"].fillna("unknown")

    # 8. review_score — intentionally left as NaN where missing.
    #    The aggregation in feature_engineering uses mean(skipna=True),
    #    so missing scores are excluded rather than distorting the average.

    dropped = original_len - len(df)
    log.info("Cleaning complete. Dropped %d rows (%.1f%%). Final shape: %s",
             dropped, 100 * dropped / original_len, df.shape)
    return df.reset_index(drop=True)


def run() -> pd.DataFrame:
    """Full pipeline: load → clean. Called by feature_engineering.py."""
    df = load_raw_data()
    return clean_data(df)


if __name__ == "__main__":
    df = run()
    print(df.dtypes)
    print(df.head(3))
