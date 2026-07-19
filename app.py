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
import os
import time
from pathlib import Path

import streamlit as st

from views import churn, copilot, customer, executive, product, regional, sales
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
        st.caption("Product Analytics · AI Insights Copilot")
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
                "✨ AI Insights Copilot",
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


# ── Setup Wizard for Deployment ────────────────────────────────────────────────

DB_PATH       = Path("data/olist.db")
FEATURES_PATH = Path("data/customer_features.parquet")


def _show_setup_page():
    st.markdown("## 📊 Welcome to Olist Analytics Dashboard")
    st.markdown("### ⚠️ Data files not found")
    st.markdown(
        "Since the database (`olist.db`) and customer feature dataset "
        "(`customer_features.parquet`) are large, they are gitignored and not "
        "stored directly in the GitHub repository."
    )
    
    st.markdown("---")
    
    st.markdown("#### 📂 Option 1: Upload Data Files (for Streamlit Cloud)")
    st.caption("Upload your local data files to quickly test the dashboard online:")
    
    os.makedirs("data", exist_ok=True)
    
    db_file = st.file_uploader("Upload olist.db", type=["db", "sqlite"])
    parquet_file = st.file_uploader("Upload customer_features.parquet", type=["parquet"])
    
    if db_file is not None:
        with open(DB_PATH, "wb") as f:
            f.write(db_file.getbuffer())
        st.success("Successfully uploaded olist.db!")
        
    if parquet_file is not None:
        with open(FEATURES_PATH, "wb") as f:
            f.write(parquet_file.getbuffer())
        st.success("Successfully uploaded customer_features.parquet!")
        
    if DB_PATH.exists() and FEATURES_PATH.exists():
        st.info("Both files uploaded successfully. Initializing dashboard…")
        time.sleep(2)
        st.rerun()

    st.markdown("---")

    st.markdown("#### 💻 Option 2: Run Locally")
    st.markdown(
        "If you want to run the application on your machine, clone the repository "
        "and run the automated setup script to download the datasets and train the ML model:"
    )
    st.code(
        "git clone https://github.com/raghavPahwa27/Product-Analytics-Dashboard.git\n"
        "cd Product-Analytics-Dashboard\n"
        "pip install -r requirements.txt\n"
        "python setup.py\n"
        "streamlit run app.py",
        language="bash",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not DB_PATH.exists() or not FEATURES_PATH.exists():
        _show_setup_page()
        return

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
