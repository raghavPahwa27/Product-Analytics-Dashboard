"""
utils/ai.py
-----------
Gemini AI client and prompt builders for the AI Business Copilot.

API key resolution order:
    1. st.secrets["GOOGLE_API_KEY"]   — Streamlit Community Cloud
    2. GOOGLE_API_KEY environment var — local / Docker
"""
import json
import os

import streamlit as st


# ── API key helper ─────────────────────────────────────────────────────────────

def get_api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.environ.get("GOOGLE_API_KEY", "")


def api_key_configured() -> bool:
    return bool(get_api_key())


# ── Core Gemini call ───────────────────────────────────────────────────────────

def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini 1.5 Flash. Returns response text or error string."""
    api_key = get_api_key()
    if not api_key:
        return (
            "⚠️  Gemini API key not configured.\n\n"
            "Set GOOGLE_API_KEY in your environment or Streamlit secrets.\n"
            "See README.md → Deployment for instructions."
        )
    try:
        import google.generativeai as genai  # noqa: PLC0415
        genai.configure(api_key=api_key)
        model    = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:
        return f"⚠️  Gemini error: {exc}"


# ── Business context builder ───────────────────────────────────────────────────

def build_business_context(df, features) -> dict:
    """
    Compute key business metrics from the order and feature datasets.
    This dict is passed as context to every AI prompt — keeping AI grounded
    in real data and preventing hallucination.
    """
    return {
        "total_revenue_BRL":              round(float(df["item_value"].sum()), 2),
        "total_orders":                   int(df["order_id"].nunique()),
        "total_unique_customers":         int(df["customer_unique_id"].nunique()),
        "avg_order_value_BRL":            round(float(df.groupby("order_id")["item_value"].sum().mean()), 2),
        "avg_review_score_out_of_5":      round(float(df["review_score"].mean()), 2),
        "churn_rate_percent":             round(float(features["churn"].mean() * 100), 1),
        "repeat_customer_rate_percent":   round(float((features["num_orders"] > 1).mean() * 100), 1),
        "avg_delivery_days":              round(float(df["delivery_days"].mean()), 1),
        "pct_orders_delayed":             round(float(df["is_delayed"].mean() * 100), 1),
        "top_5_categories_by_revenue":    (
            df.groupby("category")["item_value"].sum()
            .nlargest(5).round(2).to_dict()
        ),
        "bottom_5_categories_by_revenue": (
            df.groupby("category")["item_value"].sum()
            .nsmallest(5).round(2).to_dict()
        ),
        "top_5_states_by_revenue":        (
            df.groupby("state")["item_value"].sum()
            .nlargest(5).round(2).to_dict()
        ),
        "payment_method_order_share":     (
            df.drop_duplicates("order_id")
            .groupby("payment_type").size()
            .to_dict()
        ),
    }


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_executive_prompt(ctx: dict) -> str:
    return f"""You are a Senior Product Analyst presenting to C-suite executives.

Using ONLY the following e-commerce metrics (do not introduce external data):

{json.dumps(ctx, indent=2)}

Write a concise executive summary (4–5 paragraphs) that:
1. Summarises overall business health using the key numbers
2. Identifies the most important trend or risk visible in the data
3. Highlights a regional or product opportunity
4. Ends with 3 specific, data-driven actionable recommendations as a numbered list

Use confident business language. Synthesise numbers into insights — do not merely list them."""


def build_qa_prompt(question: str, ctx: dict) -> str:
    return f"""You are a Senior Data Analyst for an e-commerce company.

Answer the business question below using ONLY the provided metrics.
Do NOT invent numbers or reference data that is not present.
If the data is insufficient to fully answer, say so explicitly.

BUSINESS METRICS:
{json.dumps(ctx, indent=2)}

QUESTION: {question}

Give a specific, data-driven answer in 2–3 paragraphs. Reference actual figures from the data."""


def build_churn_explain_prompt(profile: dict, shap_items: list, prob: float) -> str:
    return f"""You are a Customer Success Analyst explaining a churn prediction to a business manager.

This customer has a {prob:.0%} probability of churning.

Customer Profile:
{json.dumps(profile, indent=2)}

Key prediction factors (SHAP values — positive means it increases churn risk, negative reduces it):
{json.dumps(shap_items, indent=2)}

Respond in plain business language (no ML jargon):

**Why This Customer May Churn**
In 2–3 sentences, explain the main risk signals from the factors and profile.

**Recommended Retention Actions**
List 3 specific, practical actions tailored to this customer's situation."""


def build_recommendations_prompt(ctx: dict) -> str:
    return f"""You are a Chief Product Officer reviewing this month's analytics.

Based ONLY on these business metrics:
{json.dumps(ctx, indent=2)}

Provide exactly 3 high-priority management actions. For each:
- **Action title** (bold)
- One sentence identifying the data signal driving this action
- One sentence on expected business impact

Keep it executive-level and specific to the numbers provided."""
