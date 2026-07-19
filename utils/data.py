"""
utils/data.py
-------------
Cached data loaders shared across all dashboard pages.
All heavy I/O runs once per session; subsequent calls return cached results.
"""
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── File paths ────────────────────────────────────────────────────────────────
DB_PATH       = Path("data/olist.db")
FEATURES_PATH = Path("data/customer_features.parquet")
MODEL_PATH    = Path("model/churn_model.pkl")
METRICS_PATH  = Path("model/model_metrics.json")
FI_PATH       = Path("model/feature_importance.csv")

# Columns dropped before model inference — must match train.py DROP_COLS
DROP_FOR_PRED = [
    "customer_unique_id", "city", "churn",
    "days_since_last_purchase", "days_since_first_purchase",
    "purchase_frequency",
]


@st.cache_data
def load_orders() -> pd.DataFrame:
    """Item-level delivered-orders dataset joined across all SQLite tables."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT
            o.order_id,
            strftime('%Y-%m', o.order_purchase_timestamp) AS month,
            o.order_purchase_timestamp,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            c.customer_unique_id,
            c.customer_state  AS state,
            c.customer_city   AS city,
            oi.product_id,
            oi.price,
            oi.freight_value,
            COALESCE(t.product_category_name_english,
                     pr.product_category_name, 'unknown') AS category,
            pay.payment_type,
            r.review_score
        FROM orders o
        JOIN  customers c    ON o.customer_id   = c.customer_id
        JOIN  order_items oi ON o.order_id      = oi.order_id
        LEFT JOIN products pr ON oi.product_id  = pr.product_id
        LEFT JOIN product_category_translation t
               ON pr.product_category_name      = t.product_category_name
        LEFT JOIN (
            SELECT order_id, payment_type
            FROM   payments WHERE payment_sequential = 1
        ) pay ON o.order_id = pay.order_id
        LEFT JOIN (
            SELECT order_id, AVG(review_score) AS review_score
            FROM   reviews GROUP BY order_id
        ) r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
        """,
        conn,
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    conn.close()
    df["item_value"]    = df["price"] + df["freight_value"]
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["is_delayed"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(float)
    return df


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


@st.cache_data
def load_fi() -> pd.DataFrame:
    return pd.read_csv(FI_PATH) if FI_PATH.exists() else pd.DataFrame()
