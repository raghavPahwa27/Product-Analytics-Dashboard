"""
pages/executive.py
------------------
Executive Dashboard — KPIs, monthly trends, top categories, churn split.
Includes PDF report download.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data import load_metrics
from utils.pdf import generate_report
from utils.ui import CHART_H, PALETTE, T, download_csv, heading, kpi


def render(df: pd.DataFrame, features: pd.DataFrame) -> None:
    heading(
        "🏠 Executive Dashboard",
        "High-level overview of revenue, customers, and churn",
    )

    # ── KPI strip ────────────────────────────────────────────────────────────
    rev    = df["item_value"].sum()
    orders = df["order_id"].nunique()
    cust   = df["customer_unique_id"].nunique()
    aov    = df.groupby("order_id")["item_value"].sum().mean()
    review = df["review_score"].mean()
    churn  = features["churn"].mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "💰", "Total Revenue",    f"R$ {rev:,.0f}",     "#4F8BF9")
    kpi(c2, "📦", "Total Orders",     f"{orders:,}",         "#00CC96")
    kpi(c3, "👥", "Customers",        f"{cust:,}",           "#FFA15A")
    kpi(c4, "🛒", "Avg Order Value",  f"R$ {aov:,.2f}",     "#AB63FA")
    kpi(c5, "⭐", "Avg Review Score", f"{review:.2f} / 5",  "#19D3F3")
    kpi(c6, "🔔", "Churn Rate",       f"{churn:.1f}%",       "#EF553B")

    st.divider()

    # ── Monthly revenue + orders ─────────────────────────────────────────────
    monthly_rev = df.groupby("month")["item_value"].sum().reset_index(name="revenue")
    monthly_ord = df.groupby("month")["order_id"].nunique().reset_index(name="orders")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly Revenue")
        fig = px.area(
            monthly_rev, x="month", y="revenue",
            color_discrete_sequence=[PALETTE[0]], template=T,
        )
        fig.update_traces(line_width=2)
        fig.update_layout(height=CHART_H, margin=dict(t=10, b=20),
                          showlegend=False, xaxis_title="", yaxis_title="Revenue (R$)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Monthly Orders")
        fig = px.bar(
            monthly_ord, x="month", y="orders",
            color_discrete_sequence=[PALETTE[1]], template=T,
        )
        fig.update_layout(height=CHART_H, margin=dict(t=10, b=20),
                          showlegend=False, xaxis_title="", yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)

    # ── Top categories + churn split ─────────────────────────────────────────
    top_cat = (
        df.groupby("category")["item_value"].sum()
        .nlargest(10).reset_index(name="revenue").sort_values("revenue")
    )
    churn_df = (
        features["churn"].value_counts().reset_index()
        .rename(columns={"churn": "label", "count": "n"})
    )
    churn_df["label"] = churn_df["label"].map({0: "Active", 1: "Churned"})

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top 10 Categories by Revenue")
        fig = px.bar(
            top_cat, x="revenue", y="category", orientation="h",
            color="revenue", color_continuous_scale="Blues", template=T,
        )
        fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                          margin=dict(t=10), yaxis_title="", xaxis_title="Revenue (R$)")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Customer Churn Split")
        fig = px.pie(
            churn_df, values="n", names="label",
            color="label",
            color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
            template=T,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0, 0.05])
        fig.update_layout(showlegend=False, height=CHART_H, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.divider()
    dl_col, pdf_col, _ = st.columns([1, 1, 4])

    with dl_col:
        download_csv(monthly_rev, "Monthly Revenue", "monthly_revenue.csv")

    with pdf_col:
        with st.spinner("Preparing report…"):
            pdf_bytes = generate_report(
                df, features,
                load_metrics(),
                st.session_state.get("executive_summary", ""),
            )
        if pdf_bytes:
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name="olist_executive_report.pdf",
                mime="application/pdf",
            )
