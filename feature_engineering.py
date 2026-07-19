"""
feature_engineering.py
-----------------------
Builds a one-row-per-customer feature table from the clean item-level data
produced by preprocessing.py.

Output:
    data/customer_features.csv      — human-readable, easy to inspect
    data/customer_features.parquet  — type-safe, fast I/O for ML and dashboard

Run standalone:
    python feature_engineering.py
"""

import logging
from pathlib import Path

import pandas as pd

from preprocessing import run as load_clean_data

OUTPUT_DIR           = Path("data")
CHURN_THRESHOLD_DAYS = 180   # customers inactive beyond this are labelled churned

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _item_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate at the order-item level → one row per customer.
    Covers: spend, product count, category diversity, seller diversity.
    """
    df = df.copy()
    df["item_total"] = df["price"] + df["freight_value"]

    return df.groupby("customer_unique_id").agg(
        total_spend        =("item_total",   "sum"),
        total_products     =("product_id",   "count"),
        distinct_categories=("category",     "nunique"),
        distinct_sellers   =("seller_id",    "nunique"),
        freight_sum        =("freight_value","sum"),   # divided by total_spend later to yield avg_freight_ratio
    )


def _order_level_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """
    Deduplicate to one row per order first, then aggregate per customer.
    Covers: order count, timing, delivery, review, payment, behaviour.
    """
    # One row per order — keeps the order-level columns accurate
    orders = df.drop_duplicates(subset=["order_id"]).copy()

    # Derived columns computed before groupby (vectorised)
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.days

    orders["is_delayed"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    ).astype(float)   # float so mean() gives a ratio

    orders["is_weekend"] = (
        orders["order_purchase_timestamp"].dt.dayofweek >= 5
    ).astype(float)

    agg = orders.groupby("customer_unique_id").agg(
        num_orders         =("order_id",                  "count"),
        first_purchase     =("order_purchase_timestamp",  "min"),
        last_purchase      =("order_purchase_timestamp",  "max"),
        avg_delivery_days  =("delivery_days",             "mean"),
        avg_review_score   =("review_score",              "mean"),   # NaN-safe
        pct_delayed        =("is_delayed",                "mean"),
        weekend_purchase_ratio=("is_weekend",             "mean"),
        state              =("customer_state",            "first"),
        city               =("customer_city",             "first"),
    )

    # Days since first / last purchase relative to the dataset's reference date
    agg["days_since_first_purchase"] = (reference_date - agg["first_purchase"]).dt.days
    agg["days_since_last_purchase"]  = (reference_date - agg["last_purchase"]).dt.days

    # Span of a customer's purchase history — strong churn predictor
    agg["customer_lifetime_days"] = (
        agg["last_purchase"] - agg["first_purchase"]
    ).dt.days

    return agg.drop(columns=["first_purchase", "last_purchase"])


def _preferred_payment(df: pd.DataFrame) -> pd.Series:
    """
    Return the most-used payment type per customer (vectorised, no lambda).
    Ties are broken by the first payment type alphabetically.
    """
    orders = df.drop_duplicates(subset=["order_id"])[
        ["customer_unique_id", "payment_type"]
    ]
    return (
        orders.groupby(["customer_unique_id", "payment_type"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("customer_unique_id")
        .set_index("customer_unique_id")["payment_type"]
        .rename("preferred_payment_method")
    )


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: merges item-level and order-level aggregations into a
    single customer-level DataFrame, then computes derived features.
    """
    # .normalize() truncates the time component so day-differences are whole numbers
    # and the last day of the dataset is not counted as a partial day.
    reference_date = df["order_purchase_timestamp"].max().normalize()
    log.info("Reference date (dataset end): %s", reference_date.date())

    item_agg  = _item_level_features(df)
    order_agg = _order_level_features(df, reference_date)
    payment   = _preferred_payment(df)

    features = item_agg.join(order_agg).join(payment)

    # Derived features (computed after join to avoid repeating logic)
    features["avg_order_value"]    = features["total_spend"] / features["num_orders"]
    features["avg_basket_size"]    = features["total_products"] / features["num_orders"]
    features["purchase_frequency"] = (
        features["num_orders"] / features["days_since_first_purchase"].clip(lower=1) * 30
    )  # orders per 30 days; clip(1) avoids division by zero for single-day customers
    features["avg_freight_ratio"]  = (
        features["freight_sum"] / features["total_spend"].clip(lower=0.01)
    )

    features = features.drop(columns=["freight_sum"])
    features = features.round(4)

    log.info("Built features for %d customers.", len(features))
    return features.reset_index()


def add_churn_label(features: pd.DataFrame) -> pd.DataFrame:
    """
    Churn definition: a customer is churned if they have NOT made a purchase
    within the last 180 days of the dataset window.

    Why 180 days?
    • Brazilian e-commerce average repurchase cycle is 3-6 months.
    • 180 days (6 months) is long enough to distinguish genuine inactivity
      from normal purchase intervals.
    • It is a standard industry threshold for non-subscription e-commerce.
    • Using the dataset's own end date as the reference avoids data leakage —
      we only use information available at the time of labelling.
    """
    features["churn"] = (
        features["days_since_last_purchase"] > CHURN_THRESHOLD_DAYS
    ).astype(int)

    churn_rate = features["churn"].mean()
    log.info("Churn rate: %.1f%%  (threshold: %d days)", churn_rate * 100, CHURN_THRESHOLD_DAYS)
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> pd.DataFrame:
    df       = load_clean_data()
    features = build_customer_features(df)
    features = add_churn_label(features)

    OUTPUT_DIR.mkdir(exist_ok=True)
    features.to_csv(OUTPUT_DIR / "customer_features.csv", index=False)
    features.to_parquet(OUTPUT_DIR / "customer_features.parquet", index=False)

    log.info("Saved: data/customer_features.csv")
    log.info("Saved: data/customer_features.parquet")
    return features


if __name__ == "__main__":
    features = run()
    print(f"\nShape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(features.head(3).to_string())
