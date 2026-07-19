"""
pages/regional.py
-----------------
Regional Analytics — state revenue, delivery quality, top cities.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import CHART_H, PALETTE, T, download_csv, heading


def render(df: pd.DataFrame) -> None:
    heading(
        "🌍 Regional Analytics",
        "Geographic breakdown of orders, revenue, and delivery quality",
    )

    state = df.groupby("state").agg(
        revenue     =("item_value",        "sum"),
        orders      =("order_id",          "nunique"),
        customers   =("customer_unique_id","nunique"),
        avg_delivery=("delivery_days",     "mean"),
        pct_delayed =("is_delayed",        "mean"),
    ).reset_index()

    tab_rev, tab_del, tab_city = st.tabs(
        ["💰 Revenue & Orders", "🚚 Delivery", "🏙️ Cities"]
    )

    # ── Revenue & Orders ─────────────────────────────────────────────────────
    with tab_rev:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Revenue by State")
            fig = px.bar(state.sort_values("revenue", ascending=False),
                         x="state", y="revenue", color="revenue",
                         color_continuous_scale="Blues", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), xaxis_title="", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Orders by State")
            fig = px.bar(state.sort_values("orders", ascending=False),
                         x="state", y="orders", color="orders",
                         color_continuous_scale="Greens", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), xaxis_title="", yaxis_title="Orders")
            st.plotly_chart(fig, use_container_width=True)

        download_csv(state, "State Performance", "regional_performance.csv")

    # ── Delivery ─────────────────────────────────────────────────────────────
    with tab_del:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Avg Delivery Days by State")
            fig = px.bar(state.sort_values("avg_delivery", ascending=False),
                         x="state", y="avg_delivery", color="avg_delivery",
                         color_continuous_scale="Reds", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), xaxis_title="", yaxis_title="Avg Days")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Delayed Order Rate by State")
            fig = px.bar(state.sort_values("pct_delayed", ascending=False),
                         x="state", y="pct_delayed", color="pct_delayed",
                         color_continuous_scale="Oranges", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), xaxis_title="",
                              yaxis_title="Delayed Rate",
                              yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Cities ───────────────────────────────────────────────────────────────
    with tab_city:
        cities = (
            df.groupby(["state", "city"]).agg(
                revenue=("item_value", "sum"),
                orders =("order_id",   "nunique"),
            ).reset_index().nlargest(20, "revenue")
        )
        cities["label"] = cities["city"].str.title() + " (" + cities["state"] + ")"
        st.subheader("Top 20 Cities by Revenue")
        fig = px.bar(cities.sort_values("revenue"),
                     x="revenue", y="label", orientation="h",
                     color="revenue", color_continuous_scale="Purples", template=T)
        fig.update_layout(coloraxis_showscale=False, height=560,
                          margin=dict(t=10), yaxis_title="", xaxis_title="Revenue (R$)")
        st.plotly_chart(fig, use_container_width=True)

        download_csv(cities[["state","city","revenue","orders"]],
                     "City Data", "top_cities.csv")
