"""
eda.py
------
Exploratory Data Analysis on the customer-level feature dataset.
Prints summary statistics to the console and saves Plotly charts to assets/.

Run after feature_engineering.py:
    python eda.py
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FEATURES_PATH = Path("data/customer_features.parquet")
ASSETS_DIR    = Path("assets")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    total     = len(df)
    churned   = df["churn"].sum()
    churn_pct = 100 * churned / total

    print("\n" + "=" * 50)
    print("  CUSTOMER DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total customers        : {total:,}")
    print(f"  Churned                : {churned:,}  ({churn_pct:.1f}%)")
    print(f"  Active (not churned)   : {total - churned:,}  ({100 - churn_pct:.1f}%)")
    print(f"  Avg revenue / customer : R$ {df['total_spend'].mean():,.2f}")
    print(f"  Avg orders / customer  : {df['num_orders'].mean():.2f}")
    print(f"  Avg review score       : {df['avg_review_score'].mean():.2f} / 5")
    print(f"  Avg delivery days      : {df['avg_delivery_days'].mean():.1f}")
    print("=" * 50 + "\n")


def missing_values_report(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values in customer_features.\n")
    else:
        print("Missing values:")
        print(missing.to_string())
        print()


# ---------------------------------------------------------------------------
# Plots — each function saves one HTML file to assets/
# ---------------------------------------------------------------------------

def plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df["churn"].value_counts().reset_index()
    counts.columns = ["churn", "count"]
    counts["label"] = counts["churn"].map({0: "Active", 1: "Churned"})

    fig = px.pie(
        counts, values="count", names="label",
        title="Customer Churn Distribution",
        color="label",
        color_discrete_map={"Active": "#2ecc71", "Churned": "#e74c3c"},
    )
    fig.write_html(ASSETS_DIR / "churn_distribution.html")
    log.info("Saved: assets/churn_distribution.html")


def plot_review_distribution(df: pd.DataFrame) -> None:
    # Round scores to nearest int for clean bar chart
    score_counts = (
        df["avg_review_score"].dropna()
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    score_counts.columns = ["score", "count"]

    fig = px.bar(
        score_counts, x="score", y="count",
        title="Distribution of Average Review Scores",
        labels={"score": "Review Score (1–5)", "count": "Number of Customers"},
        color="score",
        color_continuous_scale="RdYlGn",
    )
    fig.write_html(ASSETS_DIR / "review_score_distribution.html")
    log.info("Saved: assets/review_score_distribution.html")


def plot_spend_distribution(df: pd.DataFrame) -> None:
    fig = px.histogram(
        df, x="total_spend",
        nbins=80,
        title="Distribution of Customer Lifetime Spend",
        labels={"total_spend": "Total Spend (R$)"},
        log_y=True,   # log scale — spend is right-skewed
        color_discrete_sequence=["#3498db"],
    )
    fig.write_html(ASSETS_DIR / "spend_distribution.html")
    log.info("Saved: assets/spend_distribution.html")


def plot_orders_distribution(df: pd.DataFrame) -> None:
    order_counts = df["num_orders"].value_counts().sort_index().reset_index()
    order_counts.columns = ["num_orders", "customers"]

    fig = px.bar(
        order_counts.head(15), x="num_orders", y="customers",
        title="Number of Orders per Customer",
        labels={"num_orders": "Orders Placed", "customers": "Number of Customers"},
        color_discrete_sequence=["#9b59b6"],
    )
    fig.write_html(ASSETS_DIR / "orders_distribution.html")
    log.info("Saved: assets/orders_distribution.html")


def plot_churn_by_state(df: pd.DataFrame) -> None:
    state_churn = (
        df.groupby("state")["churn"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"churn": "churn_rate_pct"})
        .sort_values("churn_rate_pct", ascending=False)
        .head(15)
    )

    fig = px.bar(
        state_churn, x="state", y="churn_rate_pct",
        title="Churn Rate by State (Top 15)",
        labels={"state": "State", "churn_rate_pct": "Churn Rate (%)"},
        color="churn_rate_pct",
        color_continuous_scale="Reds",
    )
    fig.write_html(ASSETS_DIR / "churn_by_state.html")
    log.info("Saved: assets/churn_by_state.html")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    numeric_cols = [
        "num_orders", "total_spend", "avg_order_value", "avg_basket_size",
        "purchase_frequency", "days_since_first_purchase", "days_since_last_purchase",
        "avg_delivery_days", "avg_review_score", "pct_delayed", "weekend_purchase_ratio",
        "distinct_categories", "avg_freight_ratio", "churn",
    ]
    corr = df[numeric_cols].corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=700,
        xaxis_tickangle=-45,
    )
    fig.write_html(ASSETS_DIR / "correlation_matrix.html")
    log.info("Saved: assets/correlation_matrix.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Run feature_engineering.py first.")

    ASSETS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)

    print_summary(df)
    missing_values_report(df)

    plot_churn_distribution(df)
    plot_review_distribution(df)
    plot_spend_distribution(df)
    plot_orders_distribution(df)
    plot_churn_by_state(df)
    plot_correlation_matrix(df)

    log.info("EDA complete. Open any HTML file in assets/ to view charts.")


if __name__ == "__main__":
    run()
