"""
pages/product.py
----------------
Product Analytics — category revenue, bubble chart, review scores.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import CHART_H, PALETTE, T, download_csv, heading


def render(df: pd.DataFrame) -> None:
    heading(
        "📦 Product Analytics",
        "Category and product-level performance",
    )

    cat = df.groupby("category").agg(
        revenue   =("item_value",   "sum"),
        orders    =("order_id",     "nunique"),
        avg_review=("review_score", "mean"),
    ).reset_index()

    tab_top, tab_bubble, tab_review = st.tabs(
        ["🏆 Top Categories", "📈 Revenue vs Orders", "⭐ Review Scores"]
    )

    # ── Top Categories ───────────────────────────────────────────────────────
    with tab_top:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 15 by Revenue")
            top15 = cat.nlargest(15, "revenue").sort_values("revenue")
            fig = px.bar(top15, x="revenue", y="category", orientation="h",
                         color="revenue", color_continuous_scale="Blues", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), yaxis_title="", xaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Top 15 by Orders")
            top15o = cat.nlargest(15, "orders").sort_values("orders")
            fig = px.bar(top15o, x="orders", y="category", orientation="h",
                         color="orders", color_continuous_scale="Greens", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), yaxis_title="", xaxis_title="Orders")
            st.plotly_chart(fig, use_container_width=True)

        download_csv(cat.sort_values("revenue", ascending=False),
                     "Category Data", "category_performance.csv")

    # ── Bubble chart ─────────────────────────────────────────────────────────
    with tab_bubble:
        st.subheader("Revenue vs Order Volume — top 30 categories")
        top30 = cat.nlargest(30, "revenue")
        fig = px.scatter(
            top30, x="orders", y="revenue", text="category",
            size="revenue", color="avg_review",
            color_continuous_scale="RdYlGn", template=T,
            labels={"orders": "Orders", "revenue": "Revenue (R$)",
                    "avg_review": "Avg Review"},
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Review Scores ────────────────────────────────────────────────────────
    with tab_review:
        st.subheader("Highest-Rated Categories (min 50 orders)")
        rated = (
            cat[cat["orders"] >= 50]
            .dropna(subset=["avg_review"])
            .nlargest(20, "avg_review")
            .sort_values("avg_review")
        )
        fig = px.bar(rated, x="avg_review", y="category", orientation="h",
                     color="avg_review", color_continuous_scale="RdYlGn", template=T)
        fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                          margin=dict(t=10), xaxis=dict(range=[0, 5]),
                          yaxis_title="", xaxis_title="Avg Review Score")
        st.plotly_chart(fig, use_container_width=True)
