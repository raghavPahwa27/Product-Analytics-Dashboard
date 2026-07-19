"""
pages/copilot.py
----------------
AI Insights Copilot — four features powered by Groq LLaMA 3.3:
    1. Executive Summary  — full business narrative from real KPIs
    2. Ask Your Data      — question-answering grounded in live metrics
    3. Explain Customer   — SHAP-based churn explanation in plain language
    4. Recommendations    — top 3 management actions from current data
"""
import pandas as pd
import streamlit as st

from utils.ai import (
    api_key_configured,
    ask_gemini,
    build_business_context,
    build_churn_explain_prompt,
    build_executive_prompt,
    build_qa_prompt,
    build_recommendations_prompt,
)
from utils.ui import ai_box, heading


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_status_banner() -> None:
    if api_key_configured():
        st.success("✅ Groq API key configured — all features active.", icon=None)
    else:
        st.warning(
            "⚠️ Groq API key not found. "
            "Set **GROQ_API_KEY** in your environment or Streamlit secrets "
            "(see README → Deployment).",
        )


def _section(title: str) -> None:
    st.markdown(f"### {title}")
    st.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:4px 0 16px;">', unsafe_allow_html=True)


# ── Page ─────────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, features: pd.DataFrame) -> None:
    heading(
        "✨ AI Insights Copilot",
        "Real-data-grounded insights powered by Groq LLaMA 3.3",
    )
    _api_status_banner()
    st.caption(
        "Every AI response is based exclusively on your live dashboard metrics. "
        "The LLM is instructed not to introduce external data."
    )
    st.divider()

    # Pre-compute business context once per render (fast — all in-memory)
    ctx = build_business_context(df, features)

    # ── Feature 1: Executive Summary ─────────────────────────────────────────
    _section("📋 Feature 1 — Executive Summary")
    st.markdown(
        "Generates a 4–5 paragraph narrative covering performance, risks, "
        "opportunities, and 3 data-driven recommendations."
    )

    if st.button("Generate Executive Summary", key="btn_exec"):
        with st.spinner("Generating summary…"):
            summary = ask_gemini(build_executive_prompt(ctx))
        st.session_state["executive_summary"] = summary

    if "executive_summary" in st.session_state:
        ai_box(st.session_state["executive_summary"])
        st.caption("✅ This summary will be included in the PDF report (Executive page).")

    st.divider()

    # ── Feature 2: Ask Your Data ─────────────────────────────────────────────
    _section("💬 Feature 2 — Ask Your Data")
    st.markdown("Type any business question — the AI Insights Copilot answers using only your live metrics.")

    # Suggestion chips
    examples = [
        "Why is the churn rate so high?",
        "Which product categories underperform?",
        "How can we improve revenue?",
        "Which regions need attention?",
    ]
    if "qa_question" not in st.session_state:
        st.session_state["qa_question"] = ""

    chip_cols = st.columns(len(examples))
    for col, ex in zip(chip_cols, examples):
        if col.button(ex, key=f"chip_{ex[:10]}"):
            st.session_state["qa_question"] = ex

    question = st.text_input(
        "Your question",
        value=st.session_state["qa_question"],
        placeholder="e.g. Why did orders drop in Q3?",
        key="qa_input",
    )

    if st.button("Ask", key="btn_ask") and question.strip():
        with st.spinner("Thinking…"):
            answer = ask_gemini(build_qa_prompt(question.strip(), ctx))
        st.session_state["qa_answer"] = answer

    if "qa_answer" in st.session_state:
        ai_box(st.session_state["qa_answer"])

    st.divider()

    # ── Feature 3: Explain Customer Prediction ───────────────────────────────
    _section("🔍 Feature 3 — Explain Customer Prediction")
    st.markdown(
        "Select a customer to receive a plain-language explanation of why they "
        "are predicted to churn, plus tailored retention strategies. "
        "Tip: first analyse a customer on the 🤖 Churn page — "
        "their SHAP data carries over automatically."
    )

    has_shap = "last_shap" in st.session_state and "last_customer_profile" in st.session_state
    if has_shap:
        cust_id = st.session_state.get("last_customer_id", "—")[:20] + "…"
        st.info(
            f"Using data from customer `{cust_id}` (SHAP probability: "
            f"{st.session_state['last_churn_prob']:.0%}). "
            "Switch to the Churn page and select a different customer to change this."
        )

        if st.button("Explain This Prediction", key="btn_explain"):
            with st.spinner("Generating explanation…"):
                explanation = ask_gemini(
                    build_churn_explain_prompt(
                        profile   =st.session_state["last_customer_profile"],
                        shap_items=st.session_state["last_shap"],
                        prob      =st.session_state["last_churn_prob"],
                    )
                )
            st.session_state["churn_explanation"] = explanation

        if "churn_explanation" in st.session_state:
            ai_box(st.session_state["churn_explanation"])
    else:
        st.info(
            "No customer selected yet. Go to the 🤖 Churn page, "
            "pick a customer, then return here to get an AI explanation."
        )

    st.divider()

    # ── Feature 4: Business Recommendations ─────────────────────────────────
    _section("💡 Feature 4 — Business Recommendations")
    st.markdown(
        "Based on your current dashboard metrics, the AI Insights Copilot returns the "
        "top 3 management actions with data signals and expected impact."
    )

    if st.button("Get Recommendations", key="btn_rec"):
        with st.spinner("Generating recommendations…"):
            recs = ask_gemini(build_recommendations_prompt(ctx))
        st.session_state["recommendations"] = recs

    if "recommendations" in st.session_state:
        ai_box(st.session_state["recommendations"])
