"""
pages/customer.py
-----------------
Customer Analytics — spend, orders, reviews, and segment breakdown.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui import CHART_H, PALETTE, T, download_csv, heading


def render(features: pd.DataFrame) -> None:
    heading(
        "👥 Customer Analytics",
        "Spending behaviour, satisfaction, and segment breakdown",
    )

    tab_spend, tab_review, tab_seg = st.tabs(
        ["💸 Spend & Orders", "⭐ Reviews", "🔬 Segments"]
    )

    # ── Spend & Orders ───────────────────────────────────────────────────────
    with tab_spend:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Lifetime Spend Distribution")
            fig = px.histogram(features, x="total_spend", nbins=60, log_y=True,
                               color_discrete_sequence=[PALETTE[0]], template=T)
            fig.update_layout(height=CHART_H, margin=dict(t=10),
                              xaxis_title="Total Spend (R$)",
                              yaxis_title="Customers (log scale)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Orders per Customer (top 10 values)")
            od = features["num_orders"].value_counts().head(10).reset_index()
            od.columns = ["orders", "customers"]
            fig = px.bar(od, x="orders", y="customers",
                         color_discrete_sequence=[PALETTE[1]], template=T)
            fig.update_layout(height=CHART_H, margin=dict(t=10),
                              xaxis_title="Number of Orders", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)

        single = int((features["num_orders"] == 1).sum())
        repeat = int((features["num_orders"] >  1).sum())
        seg_df = pd.DataFrame({"segment": ["Single Purchase", "Repeat Buyers"],
                               "count":   [single, repeat]})
        c3, c4 = st.columns([1, 2])
        with c3:
            st.subheader("Buyer Type")
            fig = px.pie(
                seg_df, values="count", names="segment",
                color_discrete_map={"Single Purchase": PALETTE[2],
                                    "Repeat Buyers":   PALETTE[1]},
                template=T,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              pull=[0, 0.05])
            fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.subheader(" ")
            total = seg_df["count"].sum()
            for _, row in seg_df.iterrows():
                pct = row["count"] / total * 100
                st.metric(row["segment"], f"{row['count']:,}", f"{pct:.1f}% of base")

        download_csv(
            features[["customer_unique_id", "num_orders", "total_spend",
                       "avg_order_value", "avg_review_score"]],
            "Customer Summary", "customer_summary.csv",
        )

    # ── Reviews ──────────────────────────────────────────────────────────────
    with tab_review:
        scores = (
            features["avg_review_score"].dropna()
            .round().astype(int).value_counts()
            .sort_index().reset_index()
        )
        scores.columns = ["score", "count"]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Review Score Distribution")
            fig = px.bar(scores, x="score", y="count", color="score",
                         color_continuous_scale="RdYlGn", template=T)
            fig.update_layout(height=CHART_H, margin=dict(t=10),
                              coloraxis_showscale=False,
                              xaxis_title="Score (1–5)", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Avg Review: Churned vs Active")
            rv = features.groupby("churn")["avg_review_score"].mean().reset_index()
            rv["label"] = rv["churn"].map({0: "Active", 1: "Churned"})
            fig = px.bar(rv, x="label", y="avg_review_score", color="label",
                         color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
                         template=T)
            fig.update_layout(showlegend=False, height=CHART_H, margin=dict(t=10),
                              xaxis_title="", yaxis_title="Avg Review Score",
                              yaxis=dict(range=[0, 5]))
            st.plotly_chart(fig, use_container_width=True)

        delayed = features.groupby("churn")["pct_delayed"].mean().reset_index()
        delayed["label"] = delayed["churn"].map({0: "Active", 1: "Churned"})
        st.subheader("Delayed Delivery Rate: Churned vs Active")
        fig = px.bar(delayed, x="label", y="pct_delayed", color="label",
                     color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
                     template=T)
        fig.update_layout(showlegend=False, margin=dict(t=10),
                          xaxis_title="", yaxis_title="% Delayed Orders",
                          yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    # ── Segments ─────────────────────────────────────────────────────────────
    with tab_seg:
        st.subheader("Customer Segments: Spend vs Orders")
        sample = features.sample(min(3_000, len(features)), random_state=42)
        sample["Status"] = sample["churn"].map({0: "Active", 1: "Churned"})
        fig = px.scatter(
            sample, x="total_spend", y="num_orders", color="Status",
            color_discrete_map={"Active": PALETTE[1], "Churned": PALETTE[2]},
            opacity=0.45, template=T,
            labels={"total_spend": "Total Spend (R$)", "num_orders": "Orders"},
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Correlation Heatmap")
        num_cols = [
            "num_orders", "total_spend", "avg_order_value",
            "avg_delivery_days", "avg_review_score", "pct_delayed",
            "avg_freight_ratio", "customer_lifetime_days", "churn",
        ]
        corr = features[num_cols].corr().round(2)
        fig  = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0,
            text=corr.values, texttemplate="%{text}",
        ))
        fig.update_layout(template=T, height=480,
                          margin=dict(t=10), xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
