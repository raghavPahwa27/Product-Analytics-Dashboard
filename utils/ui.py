"""
utils/ui.py
-----------
Shared UI components: CSS injection, KPI cards, page headings,
chart defaults, and download helpers.
"""
import streamlit as st

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = [
    "#4F8BF9", "#00CC96", "#EF553B", "#FFA15A", "#AB63FA",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]
T       = "plotly_dark"   # Plotly template
CHART_H = 360             # standard chart height (px)

_CSS = """
<style>
/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #C9D1D9; }

/* Pull sidebar content up to remove empty top space */
div[data-testid="stSidebarUserContent"] {
    padding-top: 0.3rem !important;
}

/* ── KPI cards ────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #222840 100%);
    border: 1px solid rgba(79,139,249,0.22);
    border-radius: 14px;
    padding: 12px 10px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, linear-gradient(90deg,#4F8BF9,#7B61FF));
    border-radius: 14px 14px 0 0;
}
.kpi-icon  { font-size: 1.3rem; margin-bottom: 4px; }
.kpi-label { font-size: 0.65rem; color: #8893B0; text-transform: uppercase;
             letter-spacing: 1.1px; margin-bottom: 4px; }
.kpi-value { font-size: 1.35rem; font-weight: 700; color: #E8EBF4; line-height: 1.1; }

/* ── Risk badges ──────────────────────────────────────────────── */
.badge-high   { display:inline-block; background:#EF553B22; color:#EF553B;
                border:1px solid #EF553B55; border-radius:8px;
                padding:5px 18px; font-weight:600; font-size:1rem; }
.badge-medium { display:inline-block; background:#FFA15A22; color:#FFA15A;
                border:1px solid #FFA15A55; border-radius:8px;
                padding:5px 18px; font-weight:600; font-size:1rem; }
.badge-low    { display:inline-block; background:#00CC9622; color:#00CC96;
                border:1px solid #00CC9655; border-radius:8px;
                padding:5px 18px; font-weight:600; font-size:1rem; }

/* ── AI output box ────────────────────────────────────────────── */
.ai-box {
    background: linear-gradient(135deg, #1A1F2E 0%, #0F1419 100%);
    border: 1px solid rgba(79,139,249,0.3);
    border-radius: 12px;
    padding: 22px 26px;
    margin-top: 12px;
    line-height: 1.75;
    color: #E8EBF4;
    white-space: pre-wrap;
}

/* ── Page headings ────────────────────────────────────────────── */
.pg-title    { font-size:1.75rem; font-weight:700; color:#E8EBF4; margin-bottom:2px; }
.pg-subtitle { font-size:0.88rem; color:#8893B0; margin-bottom:20px; }

/* ── Filter pill ──────────────────────────────────────────────── */
.filter-pill {
    display:inline-block; background:#1c2433; color:#8893B0;
    border:1px solid #2d3748; border-radius:20px;
    padding:3px 12px; font-size:0.75rem; margin-right:6px; margin-bottom:8px;
}

hr { border-color: rgba(255,255,255,0.07); }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def kpi(col, icon: str, label: str, value: str, accent: str = "#4F8BF9") -> None:
    col.markdown(
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def heading(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="pg-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="pg-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def ai_box(text: str) -> None:
    """Render AI-generated text in a styled card."""
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="ai-box">{safe}</div>', unsafe_allow_html=True)


def download_csv(df, label: str, filename: str) -> None:
    st.download_button(
        label=f"⬇ Download {label} (CSV)",
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
    )


def format_metric(val: float, is_currency: bool = False) -> str:
    """Format large numbers into reader-friendly k/M strings for dashboard UI cards."""
    prefix = "R$ " if is_currency else ""
    if val >= 1_000_000:
        return f"{prefix}{val / 1_000_000:.2f}M"
    if val >= 1_000:
        return f"{prefix}{val / 1_000:.1f}k"
    if is_currency:
        return f"{prefix}{val:.2f}"
    return f"{val:,.0f}"
