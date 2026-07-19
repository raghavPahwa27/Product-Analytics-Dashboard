"""
pages/sales.py
--------------
Sales Analytics — revenue trends, product performance, payment breakdown.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui import CHART_H, PALETTE, T, download_csv, heading


def render(df: pd.DataFrame) -> None:
    heading(
        "📈 Sales Analytics",
        "Revenue trends, product performance, and payment breakdown",
    )

    tab_rev, tab_prod, tab_pay = st.tabs(["📊 Revenue", "🏷️ Products", "💳 Payments"])

    # ── Revenue tab ──────────────────────────────────────────────────────────
    with tab_rev:
        monthly = df.groupby("month").agg(
            revenue=("item_value", "sum"),
            orders =("order_id",   "nunique"),
        ).reset_index()
        monthly["growth_pct"] = monthly["revenue"].pct_change() * 100

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Monthly Revenue")
            fig = px.area(monthly, x="month", y="revenue",
                          color_discrete_sequence=[PALETTE[0]], template=T)
            fig.update_traces(line_width=2)
            fig.update_layout(height=CHART_H, margin=dict(t=10),
                              showlegend=False, xaxis_title="", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Month-over-Month Growth")
            fig = px.bar(monthly.dropna(subset=["growth_pct"]),
                         x="month", y="growth_pct", color="growth_pct",
                         color_continuous_scale="RdYlGn", template=T)
            fig.update_layout(height=CHART_H, margin=dict(t=10),
                              coloraxis_showscale=False,
                              xaxis_title="", yaxis_title="Growth (%)")
            st.plotly_chart(fig, use_container_width=True)

        top8 = (df.groupby("category")["item_value"].sum()
                .nlargest(8).index.tolist())
        cat_time = (df[df["category"].isin(top8)]
                    .groupby(["month", "category"])["item_value"]
                    .sum().reset_index(name="revenue"))
        st.subheader("Revenue by Category — Top 8 Over Time")
        fig = px.line(cat_time, x="month", y="revenue", color="category",
                      color_discrete_sequence=PALETTE, template=T)
        fig.update_layout(margin=dict(t=10), xaxis_title="",
                          yaxis_title="Revenue (R$)", legend_title="Category")
        st.plotly_chart(fig, use_container_width=True)

        download_csv(monthly, "Monthly Revenue", "monthly_revenue.csv")

    # ── Products tab ─────────────────────────────────────────────────────────
    with tab_prod:
        top15 = (df.groupby("product_id")["item_value"].sum()
                 .nlargest(15).reset_index(name="revenue").sort_values("revenue"))
        worst15 = (df.groupby("product_id")["item_value"].sum()
                   .nsmallest(15).reset_index(name="revenue")
                   .sort_values("revenue", ascending=False))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 15 Products — Revenue")
            fig = px.bar(top15, x="revenue", y="product_id", orientation="h",
                         color="revenue", color_continuous_scale="Blues", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), yaxis_title="", xaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Bottom 15 Products — Revenue")
            fig = px.bar(worst15, x="revenue", y="product_id", orientation="h",
                         color="revenue", color_continuous_scale="Reds", template=T)
            fig.update_layout(coloraxis_showscale=False, height=CHART_H,
                              margin=dict(t=10), yaxis_title="", xaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        aov_monthly = (
            df.groupby(["month", "order_id"])["item_value"].sum()
            .groupby("month").mean()
            .reset_index(name="aov")
        )
        st.subheader("Average Order Value Trend")
        fig = px.line(aov_monthly, x="month", y="aov", markers=True,
                      color_discrete_sequence=[PALETTE[3]], template=T)
        fig.update_layout(margin=dict(t=10), xaxis_title="", yaxis_title="AOV (R$)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Payments tab ─────────────────────────────────────────────────────────
    with tab_pay:
        pay = (df.drop_duplicates("order_id")
               .groupby("payment_type")
               .agg(orders=("order_id", "count"), revenue=("item_value", "sum"))
               .reset_index())

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Orders by Payment Method")
            fig = px.pie(pay, values="orders", names="payment_type",
                         color_discrete_sequence=PALETTE, template=T)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False, height=CHART_H, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Revenue by Payment Method")
            fig = px.bar(pay.sort_values("revenue", ascending=False),
                         x="payment_type", y="revenue", color="payment_type",
                         color_discrete_sequence=PALETTE, template=T)
            fig.update_layout(showlegend=False, height=CHART_H, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        download_csv(pay, "Payment Methods", "payment_breakdown.csv")
