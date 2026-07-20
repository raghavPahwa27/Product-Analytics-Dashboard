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


# ── Data paths ────────────────────────────────────────────────────────────────

DB_PATH       = Path("data/olist.db")
FEATURES_PATH = Path("data/customer_features.parquet")

# GitHub Releases URL for the large SQLite database (112 MB)
# Hosted in: https://github.com/raghavPahwa27/Product-Analytics-Dashboard/releases/tag/v1.0
DB_DOWNLOAD_URL = (
    "https://github.com/raghavPahwa27/Product-Analytics-Dashboard"
    "/releases/download/v1.0/olist.db"
)


def _download_db() -> None:
    """
    Stream-download olist.db from GitHub Releases with a live progress bar.
    Called automatically on first boot when the file is not present.
    """
    import urllib.request  # noqa: PLC0415

    os.makedirs("data", exist_ok=True)
    st.info(
        "📥 First-time setup: downloading the Olist database from GitHub Releases "
        "(~112 MB). This takes 20–40 seconds and only happens once."
    )
    progress_bar = st.progress(0, text="Starting download…")
    tmp_path = DB_PATH.with_suffix(".tmp")

    try:
        with urllib.request.urlopen(DB_DOWNLOAD_URL) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256 KB chunks

            with open(tmp_path, "wb") as out_f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(downloaded / total, 1.0)
                        mb_done = downloaded / 1_048_576
                        mb_total = total / 1_048_576
                        progress_bar.progress(
                            pct,
                            text=f"Downloading… {mb_done:.1f} / {mb_total:.1f} MB"
                        )

        tmp_path.rename(DB_PATH)
        progress_bar.progress(1.0, text="Download complete ✅")
        time.sleep(0.8)
        st.rerun()

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        st.error(f"❌ Download failed: {exc}")
        st.stop()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Auto-download olist.db from GitHub Releases if it's not present
    if not DB_PATH.exists():
        _download_db()
        return   # st.rerun() is called inside _download_db on success

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

