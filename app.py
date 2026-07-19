"""
app.py
------
Part 4: Interactive Streamlit dashboard — Product Analytics + Churn Prediction.

Pages
    🏠  Executive Dashboard  — KPIs, monthly trends, top categories
    📈  Sales Analytics      — revenue, growth, categories, products, payments
    👥  Customer Analytics   — spend, orders, reviews, segments
    📦  Product Analytics    — top / worst products, category breakdown
    🌍  Regional Analytics   — state / city revenue, delivery performance
    🤖  Churn Prediction     — live inference + SHAP individual explanation

Run:
    streamlit run app.py
"""

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

# ============================================================
# Page config  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Olist Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Paths
# ============================================================
DB_PATH       = Path("data/olist.db")
FEATURES_PATH = Path("data/customer_features.parquet")
MODEL_PATH    = Path("model/churn_model.pkl")
METRICS_PATH  = Path("model/model_metrics.json")
FI_PATH       = Path("model/feature_importance.csv")

# Columns removed before feeding a customer row to the model
# Must match train.py DROP_COLS + target column
DROP_FOR_PRED = [
    "customer_unique_id", "city", "churn",
    "days_since_last_purchase", "days_since_first_purchase", "purchase_frequency",
]

# ============================================================
# Visual tokens
# ============================================================
PALETTE = [
    "#4F8BF9", "#00CC96", "#EF553B", "#FFA15A", "#AB63FA",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]
T = "plotly_dark"   # Plotly template shorthand

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #C9D1D9; }

/* ---------- KPI cards ---------- */
.kpi-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #222840 100%);
    border: 1px solid rgba(79,139,249,0.22);
    border-radius: 14px;
    padding: 20px 16px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 6px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg,#4F8BF9,#7B61FF));
    border-radius: 14px 14px 0 0;
}
.kpi-icon  { font-size: 1.4rem; margin-bottom: 6px; }
.kpi-label { font-size: 0.68rem; color: #8893B0; text-transform: uppercase;
             letter-spacing: 1.1px; margin-bottom: 6px; }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: #E8EBF4; line-height: 1.1; }

/* ---------- Risk badges ---------- */
.badge-high   { display:inline-block; background:#EF553B22; color:#EF553B;
                border:1px solid #EF553B55; border-radius:8px;
                padding:5px 18px; font-weight:600; font-size:1rem; }
.badge-medium { display:inline-block; background:#FFA15A22; color:#FFA15A;
                border:1px solid #FFA15A55; border-radius:8px;
                padding:5px 18px; font-weight:600; font-size:1rem; }
.badge-low    { display:inline-block; background:#00CC9622; color:#00CC96;
                border:1px solid #00CC9655; border-radius:8px;
                padding:5px 18px; font-weight:600; font-size:1rem; }

/* ---------- Page heading ---------- */
.pg-title    { font-size:1.75rem; font-weight:700; color:#E8EBF4; margin-bottom:2px; }
.pg-subtitle { font-size:0.88rem; color:#8893B0; margin-bottom:20px; }

hr { border-color: rgba(255,255,255,0.07); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached data loaders
# ============================================================

@st.cache_data
def load_orders() -> pd.DataFrame:
    """Full item-level dataset (delivered orders) joined from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
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
        JOIN  customers c  ON o.customer_id  = c.customer_id
        JOIN  order_items oi ON o.order_id   = oi.order_id
        LEFT JOIN products pr ON oi.product_id = pr.product_id
        LEFT JOIN product_category_translation t
               ON pr.product_category_name = t.product_category_name
        LEFT JOIN (
            SELECT order_id, payment_type
            FROM   payments WHERE payment_sequential = 1
        ) pay ON o.order_id = pay.order_id
        LEFT JOIN (
            SELECT order_id, AVG(review_score) AS review_score
            FROM   reviews GROUP BY order_id
        ) r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
    """, conn, parse_dates=["order_purchase_timestamp",
                             "order_delivered_customer_date",
                             "order_estimated_delivery_date"])
    conn.close()
    df["item_value"]    = df["price"] + df["freight_value"]
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["is_delayed"]    = (
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


# ============================================================
# Sidebar — navigation + global filters
# ============================================================

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 📊 Olist Analytics")
        st.caption("Product Analytics · Churn Prediction")
        st.divider()

        page = st.radio(
            "nav",
            ["🏠 Executive", "📈 Sales", "👥 Customers",
             "📦 Products", "🌍 Regional", "🤖 Churn"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**🔽 Global Filters**")

        min_d = df["order_purchase_timestamp"].min().date()
        max_d = df["order_purchase_timestamp"].max().date()
        date_range = st.date_input("Date range", [min_d, max_d],
                                   min_value=min_d, max_value=max_d)

        sel_states = st.multiselect(
            "State", sorted(df["state"].dropna().unique()))
        sel_cats   = st.multiselect(
            "Category", sorted(df["category"].dropna().unique()))
        sel_pay    = st.multiselect(
            "Payment type", sorted(df["payment_type"].dropna().unique()))

        st.divider()
        st.caption("Data: Olist Brazilian E-Commerce  \nModel: XGBoost  ·  AUC 0.75")

    return page, dict(date_range=date_range, states=sel_states,
                      categories=sel_cats, payment_types=sel_pay)


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    dr = f["date_range"]
    if len(dr) == 2:
        df = df[df["order_purchase_timestamp"].dt.date.between(dr[0], dr[1])]
    if f["states"]:
        df = df[df["state"].isin(f["states"])]
    if f["categories"]:
        df = df[df["category"].isin(f["categories"])]
    if f["payment_types"]:
        df = df[df["payment_type"].isin(f["payment_types"])]
    return df


# ============================================================
# Reusable UI components
# ============================================================

def kpi(col, icon: str, label: str, value: str, accent: str = "#4F8BF9"):
    col.markdown(
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def heading(title: str, subtitle: str = ""):
    st.markdown(f'<div class="pg-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="pg-subtitle">{subtitle}</div>',
                    unsafe_allow_html=True)


# ============================================================
# Page — Executive Dashboard
# ============================================================

def page_executive(df: pd.DataFrame, features: pd.DataFrame):
    heading("🏠 Executive Dashboard",
            "High-level overview of revenue, customers and churn")

    rev    = df["item_value"].sum()
    orders = df["order_id"].nunique()
    cust   = df["customer_unique_id"].nunique()
    aov    = df.groupby("order_id")["item_value"].sum().mean()
    review = df["review_score"].mean()
    churn  = features["churn"].mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "💰", "Total Revenue",    f"R$ {rev:,.0f}",      "#4F8BF9")
    kpi(c2, "📦", "Total Orders",     f"{orders:,}",          "#00CC96")
    kpi(c3, "👥", "Customers",        f"{cust:,}",            "#FFA15A")
    kpi(c4, "🛒", "Avg Order Value",  f"R$ {aov:,.2f}",      "#AB63FA")
    kpi(c5, "⭐", "Avg Review Score", f"{review:.2f} / 5",   "#19D3F3")
    kpi(c6, "🔔", "Churn Rate",       f"{churn:.1f}%",        "#EF553B")

    st.divider()

    # Monthly revenue + orders
    c1, c2 = st.columns(2)
    monthly = (df.groupby("month")["item_value"].sum()
               .reset_index(name="revenue"))
    with c1:
        st.subheader("Monthly Revenue")
        fig = px.area(monthly, x="month", y="revenue",
                      color_discrete_sequence=[PALETTE[0]], template=T)
        fig.update_traces(line_width=2)
        fig.update_layout(margin=dict(t=10, b=20), showlegend=False,
                          xaxis_title="", yaxis_title="Revenue (R$)")
        st.plotly_chart(fig, use_container_width=True)

    mo = (df.groupby("month")["order_id"].nunique()
          .reset_index(name="orders"))
    with c2:
        st.subheader("Monthly Orders")
        fig = px.bar(mo, x="month", y="orders",
                     color_discrete_sequence=[PALETTE[1]], template=T)
        fig.update_layout(margin=dict(t=10, b=20), showlegend=False,
                          xaxis_title="", yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)

    # Top categories + churn split
    c3, c4 = st.columns(2)
    top_cat = (df.groupby("category")["item_value"].sum()
               .nlargest(10).reset_index(name="revenue")
               .sort_values("revenue"))
    with c3:
        st.subheader("Top 10 Categories by Revenue")
        fig = px.bar(top_cat, x="revenue", y="category", orientation="h",
                     color="revenue", color_continuous_scale="Blues", template=T)
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                          yaxis_title="", xaxis_title="Revenue (R$)")
        st.plotly_chart(fig, use_container_width=True)

    churn_df = (features["churn"].value_counts().reset_index()
                .rename(columns={"churn": "label", "count": "n"}))
    churn_df["label"] = churn_df["label"].map({0: "Active", 1: "Churned"})
    with c4:
        st.subheader("Customer Churn Split")
        fig = px.pie(churn_df, values="n", names="label",
                     color="label",
                     color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
                     template=T)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          pull=[0, 0.05])
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page — Sales Analytics
# ============================================================

def page_sales(df: pd.DataFrame):
    heading("📈 Sales Analytics",
            "Revenue trends, product performance and payment breakdown")

    tab_rev, tab_prod, tab_pay = st.tabs(
        ["📊 Revenue", "🏷️ Products", "💳 Payments"])

    with tab_rev:
        monthly = df.groupby("month").agg(
            revenue=("item_value", "sum"),
            orders=("order_id",   "nunique"),
        ).reset_index()
        monthly["growth_pct"] = monthly["revenue"].pct_change() * 100

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Monthly Revenue")
            fig = px.area(monthly, x="month", y="revenue",
                          color_discrete_sequence=[PALETTE[0]], template=T)
            fig.update_traces(line_width=2)
            fig.update_layout(margin=dict(t=10), showlegend=False,
                              xaxis_title="", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Month-over-Month Revenue Growth")
            fig = px.bar(monthly.dropna(subset=["growth_pct"]),
                         x="month", y="growth_pct",
                         color="growth_pct",
                         color_continuous_scale="RdYlGn", template=T)
            fig.update_layout(margin=dict(t=10), coloraxis_showscale=False,
                              xaxis_title="", yaxis_title="Growth (%)")
            st.plotly_chart(fig, use_container_width=True)

        # Category trend — top 8 only for readability
        top8 = (df.groupby("category")["item_value"].sum()
                .nlargest(8).index.tolist())
        cat_time = (df[df["category"].isin(top8)]
                    .groupby(["month", "category"])["item_value"]
                    .sum().reset_index(name="revenue"))
        st.subheader("Revenue by Category (top 8) Over Time")
        fig = px.line(cat_time, x="month", y="revenue", color="category",
                      color_discrete_sequence=PALETTE, template=T)
        fig.update_layout(margin=dict(t=10),
                          xaxis_title="", yaxis_title="Revenue (R$)",
                          legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)

    with tab_prod:
        c1, c2 = st.columns(2)
        top15 = (df.groupby("product_id")["item_value"].sum()
                 .nlargest(15).reset_index(name="revenue")
                 .sort_values("revenue"))
        with c1:
            st.subheader("Top 15 Products — Revenue")
            fig = px.bar(top15, x="revenue", y="product_id", orientation="h",
                         color="revenue", color_continuous_scale="Blues", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              yaxis_title="", xaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        worst15 = (df.groupby("product_id")["item_value"].sum()
                   .nsmallest(15).reset_index(name="revenue")
                   .sort_values("revenue", ascending=False))
        with c2:
            st.subheader("Bottom 15 Products — Revenue")
            fig = px.bar(worst15, x="revenue", y="product_id", orientation="h",
                         color="revenue", color_continuous_scale="Reds", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              yaxis_title="", xaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        aov_monthly = (
            df.groupby(["month", "order_id"])["item_value"].sum()
            .groupby("month").mean()
            .reset_index(name="aov")
        )
        st.subheader("Average Order Value Trend")
        fig = px.line(aov_monthly, x="month", y="aov", markers=True,
                      color_discrete_sequence=[PALETTE[3]], template=T)
        fig.update_layout(margin=dict(t=10),
                          xaxis_title="", yaxis_title="AOV (R$)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_pay:
        pay = (df.drop_duplicates("order_id")
               .groupby("payment_type")
               .agg(orders=("order_id",  "count"),
                    revenue=("item_value","sum"))
               .reset_index())

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Orders by Payment Method")
            fig = px.pie(pay, values="orders", names="payment_type",
                         color_discrete_sequence=PALETTE, template=T)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Revenue by Payment Method")
            fig = px.bar(pay.sort_values("revenue", ascending=False),
                         x="payment_type", y="revenue", color="payment_type",
                         color_discrete_sequence=PALETTE, template=T)
            fig.update_layout(showlegend=False, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page — Customer Analytics
# ============================================================

def page_customers(features: pd.DataFrame):
    heading("👥 Customer Analytics",
            "Spending behaviour, satisfaction and segment breakdown")

    tab_spend, tab_review, tab_seg = st.tabs(
        ["💸 Spend & Orders", "⭐ Reviews", "🔬 Segments"])

    with tab_spend:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Lifetime Spend Distribution")
            fig = px.histogram(features, x="total_spend", nbins=60, log_y=True,
                               color_discrete_sequence=[PALETTE[0]], template=T)
            fig.update_layout(margin=dict(t=10),
                              xaxis_title="Total Spend (R$)",
                              yaxis_title="Customers (log scale)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Orders per Customer (top 10)")
            od = (features["num_orders"].value_counts()
                  .head(10).reset_index())
            od.columns = ["orders", "customers"]
            fig = px.bar(od, x="orders", y="customers",
                         color_discrete_sequence=[PALETTE[1]], template=T)
            fig.update_layout(margin=dict(t=10),
                              xaxis_title="Number of Orders",
                              yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)

        # Single vs repeat
        single = (features["num_orders"] == 1).sum()
        repeat = (features["num_orders"]  > 1).sum()
        seg_df = pd.DataFrame({
            "segment": ["Single Purchase", "Repeat Buyers"],
            "count":   [single, repeat],
        })
        c3, c4 = st.columns([1, 2])
        with c3:
            st.subheader("Buyer Type")
            fig = px.pie(seg_df, values="count", names="segment",
                         color_discrete_map={
                             "Single Purchase": PALETTE[2],
                             "Repeat Buyers":   PALETTE[1],
                         }, template=T)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              pull=[0, 0.05])
            fig.update_layout(showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.subheader(" ")
            for _, row in seg_df.iterrows():
                pct = row["count"] / seg_df["count"].sum() * 100
                st.metric(row["segment"], f"{row['count']:,}", f"{pct:.1f}% of base")

    with tab_review:
        c1, c2 = st.columns(2)
        scores = (features["avg_review_score"].dropna()
                  .round().astype(int).value_counts()
                  .sort_index().reset_index())
        scores.columns = ["score", "count"]
        with c1:
            st.subheader("Review Score Distribution")
            fig = px.bar(scores, x="score", y="count",
                         color="score", color_continuous_scale="RdYlGn", template=T)
            fig.update_layout(margin=dict(t=10), coloraxis_showscale=False,
                              xaxis_title="Score (1–5)", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Avg Review: Churned vs Active")
            rv = (features.groupby("churn")["avg_review_score"]
                  .mean().reset_index())
            rv["label"] = rv["churn"].map({0: "Active", 1: "Churned"})
            fig = px.bar(rv, x="label", y="avg_review_score",
                         color="label",
                         color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
                         template=T)
            fig.update_layout(showlegend=False, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Avg Review Score",
                              yaxis=dict(range=[0, 5]))
            st.plotly_chart(fig, use_container_width=True)

        delayed = (features.groupby("churn")["pct_delayed"]
                   .mean().reset_index())
        delayed["label"] = delayed["churn"].map({0: "Active", 1: "Churned"})
        st.subheader("Delayed Delivery Rate: Churned vs Active")
        fig = px.bar(delayed, x="label", y="pct_delayed",
                     color="label",
                     color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
                     template=T)
        fig.update_layout(showlegend=False, margin=dict(t=10),
                          xaxis_title="", yaxis_title="% Delayed Orders",
                          yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    with tab_seg:
        st.subheader("Customer Segments: Spend vs Orders")
        sample = features.sample(min(3_000, len(features)), random_state=42)
        sample["churn_label"] = sample["churn"].map({0: "Active", 1: "Churned"})
        fig = px.scatter(sample, x="total_spend", y="num_orders",
                         color="churn_label",
                         color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
                         opacity=0.45, template=T,
                         labels={"total_spend": "Total Spend (R$)",
                                 "num_orders":  "Orders",
                                 "churn_label": "Status"})
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Correlation Heatmap")
        num_cols = ["num_orders", "total_spend", "avg_order_value",
                    "avg_delivery_days", "avg_review_score", "pct_delayed",
                    "avg_freight_ratio", "customer_lifetime_days", "churn"]
        corr = features[num_cols].corr().round(2)
        fig  = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0,
            text=corr.values, texttemplate="%{text}",
        ))
        fig.update_layout(template=T, height=480, margin=dict(t=10),
                          xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page — Product Analytics
# ============================================================

def page_products(df: pd.DataFrame):
    heading("📦 Product Analytics",
            "Category and product-level performance")

    cat = df.groupby("category").agg(
        revenue   =("item_value",  "sum"),
        orders    =("order_id",    "nunique"),
        avg_review=("review_score","mean"),
    ).reset_index()

    tab_top, tab_bubble, tab_review = st.tabs(
        ["🏆 Top Categories", "📈 Revenue vs Orders", "⭐ Review Scores"])

    with tab_top:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 15 by Revenue")
            top15 = cat.nlargest(15, "revenue").sort_values("revenue")
            fig = px.bar(top15, x="revenue", y="category", orientation="h",
                         color="revenue", color_continuous_scale="Blues", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              yaxis_title="", xaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top 15 by Orders")
            top15o = cat.nlargest(15, "orders").sort_values("orders")
            fig = px.bar(top15o, x="orders", y="category", orientation="h",
                         color="orders", color_continuous_scale="Greens", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              yaxis_title="", xaxis_title="Orders")
            st.plotly_chart(fig, use_container_width=True)

    with tab_bubble:
        st.subheader("Revenue vs Order Volume — top 30 categories")
        top30 = cat.nlargest(30, "revenue")
        fig = px.scatter(top30, x="orders", y="revenue", text="category",
                         size="revenue", color="avg_review",
                         color_continuous_scale="RdYlGn", template=T,
                         labels={"orders": "Orders", "revenue": "Revenue (R$)",
                                 "avg_review": "Avg Review"})
        fig.update_traces(textposition="top center")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_review:
        st.subheader("Highest-Rated Categories (min 50 orders)")
        rated = (cat[cat["orders"] >= 50]
                 .dropna(subset=["avg_review"])
                 .nlargest(20, "avg_review")
                 .sort_values("avg_review"))
        fig = px.bar(rated, x="avg_review", y="category", orientation="h",
                     color="avg_review", color_continuous_scale="RdYlGn", template=T)
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                          xaxis=dict(range=[0, 5]), yaxis_title="",
                          xaxis_title="Avg Review Score")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page — Regional Analytics
# ============================================================

def page_regional(df: pd.DataFrame):
    heading("🌍 Regional Analytics",
            "Geographic breakdown of orders, revenue and delivery quality")

    state = df.groupby("state").agg(
        revenue     =("item_value",   "sum"),
        orders      =("order_id",     "nunique"),
        customers   =("customer_unique_id", "nunique"),
        avg_delivery=("delivery_days","mean"),
        pct_delayed =("is_delayed",   "mean"),
    ).reset_index()

    tab_rev, tab_del, tab_city = st.tabs(
        ["💰 Revenue & Orders", "🚚 Delivery", "🏙️ Cities"])

    with tab_rev:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Revenue by State")
            fig = px.bar(state.sort_values("revenue", ascending=False),
                         x="state", y="revenue",
                         color="revenue", color_continuous_scale="Blues", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Orders by State")
            fig = px.bar(state.sort_values("orders", ascending=False),
                         x="state", y="orders",
                         color="orders", color_continuous_scale="Greens", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Orders")
            st.plotly_chart(fig, use_container_width=True)

    with tab_del:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Avg Delivery Days by State")
            fig = px.bar(state.sort_values("avg_delivery", ascending=False),
                         x="state", y="avg_delivery",
                         color="avg_delivery", color_continuous_scale="Reds", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Avg Delivery Days")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Delayed Order Rate by State")
            fig = px.bar(state.sort_values("pct_delayed", ascending=False),
                         x="state", y="pct_delayed",
                         color="pct_delayed", color_continuous_scale="Oranges", template=T)
            fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Delayed Rate",
                              yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

    with tab_city:
        cities = (df.groupby(["state", "city"]).agg(
            revenue=("item_value", "sum"),
            orders =("order_id",   "nunique"),
        ).reset_index().nlargest(20, "revenue"))
        cities["label"] = cities["city"].str.title() + " (" + cities["state"] + ")"
        st.subheader("Top 20 Cities by Revenue")
        fig = px.bar(cities.sort_values("revenue"),
                     x="revenue", y="label", orientation="h",
                     color="revenue", color_continuous_scale="Purples", template=T)
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                          yaxis_title="", xaxis_title="Revenue (R$)")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page — Churn Prediction
# ============================================================

def _shap_values(pipeline, row_df: pd.DataFrame):
    """
    Compute SHAP values for a single customer using the fitted pipeline.
    Returns (shap_array, feature_names) or (None, None) on failure.
    """
    try:
        import shap
        clf       = pipeline.named_steps["clf"]
        pre_steps = pipeline.steps[:-1]
        X_trans   = Pipeline(pre_steps).transform(row_df)
        names     = np.array([
            n.split("__", 1)[-1]
            for n in pipeline.named_steps["pre"].get_feature_names_out()
        ])
        explainer = shap.TreeExplainer(clf)
        sv        = explainer.shap_values(X_trans)
        if isinstance(sv, list):   # RandomForest returns list
            sv = sv[1]
        return sv[0], names
    except Exception:
        return None, None


def page_churn(features: pd.DataFrame):
    heading("🤖 Churn Prediction",
            "Live inference · SHAP individual explanation · business action")

    model = load_model()
    if model is None:
        st.error("Model not found. Run `python train.py` first.")
        return

    # ---- Model performance summary bar
    metrics = load_metrics()
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "🏆", "Model",      metrics.get("model", "—"),             "#4F8BF9")
        kpi(c2, "📊", "ROC AUC",    f"{metrics.get('roc_auc', 0):.4f}",   "#00CC96")
        kpi(c3, "🎯", "F1",         f"{metrics.get('f1', 0):.4f}",         "#FFA15A")
        kpi(c4, "🔍", "Precision",  f"{metrics.get('precision', 0):.4f}",  "#AB63FA")
        kpi(c5, "📡", "Recall",     f"{metrics.get('recall', 0):.4f}",     "#19D3F3")
        st.divider()

    col_left, col_right = st.columns([1, 2])

    # ---- Left: customer selector + profile
    with col_left:
        st.subheader("🔍 Select Customer")
        ids = features["customer_unique_id"].head(300).tolist()
        sel_id = st.selectbox(
            "Customer ID",
            ids,
            format_func=lambda x: x[:20] + "…",
        )
        cust = features[features["customer_unique_id"] == sel_id].iloc[0]

        with st.expander("📋 Customer Profile", expanded=True):
            show_cols = [
                "num_orders", "total_spend", "avg_order_value",
                "avg_review_score", "pct_delayed", "avg_delivery_days",
                "customer_lifetime_days", "state", "preferred_payment_method",
            ]
            for col in show_cols:
                val = cust.get(col, "—")
                label = col.replace("_", " ").title()
                if isinstance(val, float):
                    st.metric(label, f"{val:.2f}")
                else:
                    st.metric(label, str(val))

    # ---- Right: prediction + SHAP
    with col_right:
        drop_present = [c for c in DROP_FOR_PRED if c in cust.index]
        input_df     = pd.DataFrame([cust.drop(drop_present)])
        prob         = float(model.predict_proba(input_df)[0, 1])

        # Gauge
        st.subheader("Churn Probability")
        needle_color = (PALETTE[2] if prob > 0.6
                        else PALETTE[3] if prob > 0.35 else PALETTE[1])
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#E8EBF4"}},
            gauge={
                "axis":      {"range": [0, 100], "tickcolor": "#8893B0"},
                "bar":       {"color": needle_color},
                "steps": [
                    {"range": [0,  35],  "color": "#0E2A1A"},
                    {"range": [35, 60],  "color": "#2A200A"},
                    {"range": [60, 100], "color": "#2A0A0A"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.8, "value": prob * 100},
            },
        ))
        fig.update_layout(
            template=T, height=260, margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Risk badge + recommendation
        if prob > 0.6:
            badge = "badge-high"
            label = "🔴 High Risk"
            rec   = ("**Immediate action recommended.** Send a personalised "
                     "discount or loyalty-points offer within 48 hours. "
                     "Consider a win-back email highlighting new arrivals in "
                     "their most-purchased category.")
        elif prob > 0.35:
            badge = "badge-medium"
            label = "🟠 Medium Risk"
            rec   = ("**Monitor and nudge.** Trigger a light re-engagement "
                     "email with product recommendations. Consider a small "
                     "free-shipping voucher for next order.")
        else:
            badge = "badge-low"
            label = "🟢 Low Risk"
            rec   = ("**Customer appears healthy.** No immediate action needed. "
                     "Focus retention budget on higher-risk segments.")

        st.markdown(f'<span class="{badge}">{label}</span>',
                    unsafe_allow_html=True)
        st.caption("")
        with st.expander("💡 Business Recommendation", expanded=True):
            st.markdown(rec)

        # SHAP waterfall
        sv, names = _shap_values(model, input_df)
        if sv is not None:
            shap_df = (pd.DataFrame({"feature": names, "shap": sv})
                       .assign(abs_shap=lambda d: d["shap"].abs())
                       .nlargest(12, "abs_shap")
                       .sort_values("shap"))
            colors = [PALETTE[2] if v > 0 else PALETTE[1]
                      for v in shap_df["shap"]]
            fig = go.Figure(go.Bar(
                x=shap_df["shap"], y=shap_df["feature"],
                orientation="h", marker_color=colors,
            ))
            fig.update_layout(
                template=T,
                title="Why this prediction? (SHAP values)",
                xaxis_title="Impact on churn probability  →  positive = more churn",
                margin=dict(t=40, b=10),
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---- Global feature importance
    st.divider()
    fi = load_fi()
    if not fi.empty:
        st.subheader("📊 Global Feature Importances (XGBoost)")
        top15 = fi.head(15).sort_values("importance")
        fig = px.bar(top15, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="Blues", template=T)
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                          yaxis_title="", xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 Classification Report"):
            cr_path = Path("model/classification_report.txt")
            if cr_path.exists():
                st.code(cr_path.read_text(), language=None)


# ============================================================
# Main
# ============================================================

def main():
    if not DB_PATH.exists():
        st.error("Database not found. Run `python database.py` first.")
        st.stop()
    if not FEATURES_PATH.exists():
        st.error("Features not found. Run `python feature_engineering.py` first.")
        st.stop()

    df       = load_orders()
    features = load_features()

    page, filters = render_sidebar(df)
    fdf = apply_filters(df, filters)

    if   page.startswith("🏠"): page_executive(fdf, features)
    elif page.startswith("📈"): page_sales(fdf)
    elif page.startswith("👥"): page_customers(features)
    elif page.startswith("📦"): page_products(fdf)
    elif page.startswith("🌍"): page_regional(fdf)
    elif page.startswith("🤖"): page_churn(features)


if __name__ == "__main__":
    main()
