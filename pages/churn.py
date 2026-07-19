"""
pages/churn.py
--------------
Churn Prediction — live inference, SHAP individual explanation, risk badge,
business recommendation, and global feature importance.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

from utils.data import DROP_FOR_PRED, load_fi, load_metrics, load_model
from utils.ui import CHART_H, PALETTE, T, heading, kpi


# ── SHAP helper ───────────────────────────────────────────────────────────────

def _shap_values(pipeline, row_df: pd.DataFrame):
    """
    Compute SHAP values for a single row using the fitted pipeline.
    Returns (shap_array, feature_names) or (None, None) on failure.
    """
    try:
        import shap  # noqa: PLC0415
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


# ── Page ─────────────────────────────────────────────────────────────────────

def render(features: pd.DataFrame) -> None:
    heading(
        "🤖 Churn Prediction",
        "Live inference · SHAP individual explanation · business action",
    )

    model = load_model()
    if model is None:
        st.error("Model not found — run `python train.py` first.")
        return

    # ── Model KPI strip ──────────────────────────────────────────────────────
    metrics = load_metrics()
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "🏆", "Model",     metrics.get("model", "—"),            "#4F8BF9")
        kpi(c2, "📊", "ROC AUC",   f"{metrics.get('roc_auc', 0):.4f}",  "#00CC96")
        kpi(c3, "🎯", "F1",        f"{metrics.get('f1', 0):.4f}",        "#FFA15A")
        kpi(c4, "🔍", "Precision", f"{metrics.get('precision', 0):.4f}", "#AB63FA")
        kpi(c5, "📡", "Recall",    f"{metrics.get('recall', 0):.4f}",    "#19D3F3")
        st.divider()

    col_left, col_right = st.columns([1, 2])

    # ── Customer selector ────────────────────────────────────────────────────
    with col_left:
        st.subheader("🔍 Select Customer")
        ids    = features["customer_unique_id"].head(300).tolist()
        sel_id = st.selectbox("Customer ID", ids,
                              format_func=lambda x: x[:20] + "…")
        cust   = features[features["customer_unique_id"] == sel_id].iloc[0]

        with st.expander("📋 Customer Profile", expanded=True):
            show_cols = [
                "num_orders", "total_spend", "avg_order_value",
                "avg_review_score", "pct_delayed", "avg_delivery_days",
                "customer_lifetime_days", "state", "preferred_payment_method",
            ]
            for col in show_cols:
                val   = cust.get(col, "—")
                label = col.replace("_", " ").title()
                st.metric(label, f"{val:.2f}" if isinstance(val, float) else str(val))

    # ── Prediction + SHAP ────────────────────────────────────────────────────
    with col_right:
        drop_present = [c for c in DROP_FOR_PRED if c in cust.index]
        input_df     = pd.DataFrame([cust.drop(drop_present)])
        prob         = float(model.predict_proba(input_df)[0, 1])

        # Gauge
        st.subheader("Churn Probability")
        needle_color = PALETTE[2] if prob > 0.6 else PALETTE[3] if prob > 0.35 else PALETTE[1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#E8EBF4"}},
            gauge={
                "axis":      {"range": [0, 100], "tickcolor": "#8893B0"},
                "bar":       {"color": needle_color},
                "steps":     [
                    {"range": [0,  35], "color": "#0E2A1A"},
                    {"range": [35, 60], "color": "#2A200A"},
                    {"range": [60,100], "color": "#2A0A0A"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.8, "value": prob * 100},
            },
        ))
        fig.update_layout(template=T, height=260, margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Risk badge + recommendation
        if prob > 0.6:
            badge, label = "badge-high", "🔴 High Risk"
            rec = (
                "**Immediate action recommended.** Send a personalised discount or "
                "loyalty-points offer within 48 hours. Highlight new arrivals in their "
                "most-purchased category."
            )
        elif prob > 0.35:
            badge, label = "badge-medium", "🟠 Medium Risk"
            rec = (
                "**Monitor and nudge.** Trigger a re-engagement email with product "
                "recommendations. Consider a small free-shipping voucher for next order."
            )
        else:
            badge, label = "badge-low", "🟢 Low Risk"
            rec = (
                "**Customer appears healthy.** No immediate action needed. "
                "Focus retention budget on higher-risk segments."
            )

        st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
        st.caption("")
        with st.expander("💡 Business Recommendation", expanded=True):
            st.markdown(rec)

        # SHAP waterfall
        sv, names = _shap_values(model, input_df)
        if sv is not None:
            shap_df = (
                pd.DataFrame({"feature": names, "shap": sv})
                .assign(abs_shap=lambda d: d["shap"].abs())
                .nlargest(12, "abs_shap")
                .sort_values("shap")
            )
            colors = [PALETTE[2] if v > 0 else PALETTE[1] for v in shap_df["shap"]]
            fig = go.Figure(go.Bar(
                x=shap_df["shap"], y=shap_df["feature"],
                orientation="h", marker_color=colors,
            ))
            fig.update_layout(
                template=T,
                title="Why this prediction? (SHAP values)",
                xaxis_title="Impact on churn probability  →  positive = more churn",
                margin=dict(t=40, b=10), height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Store SHAP data in session state for the AI Copilot page
        if sv is not None:
            shap_items = (
                pd.DataFrame({"feature": names, "shap": sv})
                .assign(abs_shap=lambda d: d["shap"].abs())
                .nlargest(8, "abs_shap")
                .sort_values("shap", ascending=False)
                [["feature", "shap"]]
                .round(4)
                .to_dict("records")
            )
            st.session_state["last_shap"]         = shap_items
            st.session_state["last_churn_prob"]   = prob
            st.session_state["last_customer_id"]  = sel_id
            profile_keys = [
                "num_orders", "total_spend", "avg_order_value",
                "avg_review_score", "pct_delayed", "avg_delivery_days",
                "customer_lifetime_days", "state", "preferred_payment_method",
            ]
            st.session_state["last_customer_profile"] = {
                k: (round(float(cust[k]), 3) if isinstance(cust.get(k), float) else str(cust.get(k, "—")))
                for k in profile_keys if k in cust.index
            }

    # ── Global feature importance ────────────────────────────────────────────
    st.divider()
    fi = load_fi()
    if not fi.empty:
        st.subheader("📊 Global Feature Importances (XGBoost)")
        top15 = fi.head(15).sort_values("importance")
        fig = go.Figure(go.Bar(
            x=top15["importance"], y=top15["feature"],
            orientation="h",
            marker=dict(
                color=top15["importance"],
                colorscale="Blues",
                showscale=False,
            ),
        ))
        fig.update_layout(template=T, height=CHART_H, margin=dict(t=10),
                          xaxis_title="Importance Score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 Classification Report"):
            from pathlib import Path  # noqa: PLC0415
            cr_path = Path("model/classification_report.txt")
            if cr_path.exists():
                st.code(cr_path.read_text(), language=None)

    st.caption("💡 Navigate to ✨ AI Copilot to get a plain-English explanation of this prediction.")
