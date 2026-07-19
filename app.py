"""
app.py
------
Olist Product Analytics Dashboard — main entry point.

Responsibilities:
    - Streamlit page configuration
    - CSS injection
    - Sidebar: navigation + global filters
    - Route to the appropriate page module

Run:
    streamlit run app.py
"""
import streamlit as st

from pages import churn, copilot, customer, executive, product, regional, sales
from utils.data import load_features, load_orders
from utils.ui import inject_css

# ── Page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Olist Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar(df):
    with st.sidebar:
        st.markdown("## 📊 Olist Analytics")
        st.caption("Product Analytics · AI Business Copilot")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🏠 Executive",
                "📈 Sales",
                "👥 Customers",
                "📦 Products",
                "🌍 Regional",
                "🤖 Churn",
                "✨ AI Copilot",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**🔽 Global Filters**")
        st.caption("Applied to order-level pages (not Churn / Copilot).")

        min_d = df["order_purchase_timestamp"].min().date()
        max_d = df["order_purchase_timestamp"].max().date()
        date_range = st.date_input("Date range", [min_d, max_d],
                                   min_value=min_d, max_value=max_d)

        sel_states = st.multiselect("State",        sorted(df["state"].dropna().unique()))
        sel_cats   = st.multiselect("Category",     sorted(df["category"].dropna().unique()))
        sel_pay    = st.multiselect("Payment type", sorted(df["payment_type"].dropna().unique()))

        st.divider()
        st.caption("Data: Olist Brazilian E-Commerce\nModel: XGBoost · AUC 0.75\nv1.5.0")

    return page, dict(
        date_range=date_range,
        states=sel_states,
        categories=sel_cats,
        payment_types=sel_pay,
    )


def _apply_filters(df, f):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Loading data…"):
        df       = load_orders()
        features = load_features()

    page, filters = _sidebar(df)
    fdf = _apply_filters(df, filters)

    if   page.startswith("🏠"): executive.render(fdf, features)
    elif page.startswith("📈"): sales.render(fdf)
    elif page.startswith("👥"): customer.render(features)
    elif page.startswith("📦"): product.render(fdf)
    elif page.startswith("🌍"): regional.render(fdf)
    elif page.startswith("🤖"): churn.render(features)
    elif page.startswith("✨"): copilot.render(fdf, features)


if __name__ == "__main__":
    main()
