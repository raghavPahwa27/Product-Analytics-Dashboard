"""
utils/pdf.py
------------
ReportLab PDF report generator.
Returns the report as bytes — suitable for st.download_button.
No files are written to disk.
"""
from datetime import date
from io import BytesIO

import pandas as pd


def generate_report(
    df: pd.DataFrame,
    features: pd.DataFrame,
    ml_metrics: dict,
    ai_summary: str = "",
) -> bytes:
    """
    Build an A4 executive report and return it as bytes.
    Returns b"" if reportlab is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable, PageBreak, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        return b""

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    NAVY   = colors.HexColor("#1a3a6b")
    BLUE   = colors.HexColor("#2c5282")
    ACCENT = colors.HexColor("#4F8BF9")
    LTBLUE = colors.HexColor("#ebf4ff")

    title_s = ParagraphStyle(
        "T2", parent=styles["Title"],
        textColor=NAVY, fontSize=28, spaceAfter=6,
    )
    h1_s = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        textColor=NAVY, fontSize=16, spaceBefore=18, spaceAfter=6,
    )
    h2_s = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        textColor=BLUE, fontSize=12, spaceBefore=10, spaceAfter=4,
    )
    body_s = ParagraphStyle(
        "B2", parent=styles["Normal"],
        fontSize=10, leading=15,
    )
    caption_s = ParagraphStyle(
        "Cap", parent=styles["Normal"],
        fontSize=8, textColor=colors.grey,
    )

    def hr():
        return HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8)

    def tbl(data, widths=None):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1,  0), NAVY),
            ("TEXTCOLOR",      (0, 0), (-1,  0), colors.white),
            ("FONTNAME",       (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1,  0), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LTBLUE]),
            ("FONTSIZE",       (0, 1), (-1, -1), 9),
            ("GRID",           (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ]))
        return t

    sp   = lambda n=1: Spacer(1, n * 0.4 * cm)
    para = lambda text, style=body_s: Paragraph(text, style)

    story = []

    # ── Cover ───────────────────────────────────────────────────────────────
    story += [
        sp(6),
        para("Olist Analytics", title_s),
        para("Executive Business Report", h1_s),
        sp(),
        para(f"Generated: {date.today().strftime('%d %B %Y')}", caption_s),
        sp(),
        hr(),
        para(
            "This report summarises key performance indicators, customer behaviour, "
            "product insights, regional performance, and churn model results "
            "for the Olist Brazilian e-commerce platform.",
        ),
        PageBreak(),
    ]

    # ── 1. Revenue & Orders ──────────────────────────────────────────────────
    rev     = df["item_value"].sum()
    orders  = df["order_id"].nunique()
    cust    = df["customer_unique_id"].nunique()
    aov     = df.groupby("order_id")["item_value"].sum().mean()
    review  = df["review_score"].mean()
    churn   = features["churn"].mean() * 100
    repeat  = (features["num_orders"] > 1).mean() * 100
    delayed = df["is_delayed"].mean() * 100

    story += [
        para("1. Revenue &amp; Order Summary", h1_s), hr(),
        tbl(
            [
                ["Metric",                    "Value"],
                ["Total Revenue (R$)",         f"R$ {rev:,.2f}"],
                ["Total Orders",               f"{orders:,}"],
                ["Unique Customers",           f"{cust:,}"],
                ["Average Order Value (R$)",   f"R$ {aov:,.2f}"],
                ["Average Review Score",       f"{review:.2f} / 5.00"],
                ["Customer Churn Rate",        f"{churn:.1f}%"],
                ["Repeat Customer Rate",       f"{repeat:.1f}%"],
                ["Orders Delivered Late",      f"{delayed:.1f}%"],
            ],
            widths=[10 * cm, 6 * cm],
        ),
        sp(),
    ]

    monthly = df.groupby("month")["item_value"].sum()
    best_m  = monthly.idxmax()
    worst_m = monthly.idxmin()
    story  += [
        para(
            f"Best month: <b>{best_m}</b> (R$ {monthly[best_m]:,.0f}) &nbsp;|&nbsp; "
            f"Lowest month: <b>{worst_m}</b> (R$ {monthly[worst_m]:,.0f})"
        ),
        PageBreak(),
    ]

    # ── 2. Top Categories ────────────────────────────────────────────────────
    top_cats = (
        df.groupby("category")["item_value"].sum()
        .nlargest(10).reset_index(name="revenue")
    )
    cat_rows = [["Rank", "Category", "Revenue (R$)"]] + [
        [str(i + 1), row["category"], f"R$ {row['revenue']:,.2f}"]
        for i, (_, row) in enumerate(top_cats.iterrows())
    ]
    story += [
        para("2. Product &amp; Category Performance", h1_s), hr(),
        para("Top 10 Categories by Revenue", h2_s),
        tbl(cat_rows, widths=[2 * cm, 9 * cm, 5 * cm]),
        sp(),
        PageBreak(),
    ]

    # ── 3. Regional Performance ──────────────────────────────────────────────
    state_agg = (
        df.groupby("state").agg(
            Revenue    =("item_value",   "sum"),
            Orders     =("order_id",     "nunique"),
            AvgDeliv   =("delivery_days","mean"),
            PctDelayed =("is_delayed",   "mean"),
        )
        .nlargest(10, "Revenue")
        .reset_index()
    )
    reg_rows = [["State", "Revenue (R$)", "Orders", "Avg Delivery", "% Delayed"]] + [
        [
            r["state"],
            f"R$ {r['Revenue']:,.0f}",
            f"{r['Orders']:,}",
            f"{r['AvgDeliv']:.1f} d",
            f"{r['PctDelayed'] * 100:.1f}%",
        ]
        for _, r in state_agg.iterrows()
    ]
    story += [
        para("3. Regional Performance", h1_s), hr(),
        tbl(reg_rows, widths=[2.5*cm, 4.5*cm, 3*cm, 3.5*cm, 3.5*cm]),
        sp(),
        PageBreak(),
    ]

    # ── 4. Churn Model ───────────────────────────────────────────────────────
    story += [
        para("4. Customer Churn Model", h1_s), hr(),
        para(
            "The churn model uses XGBoost trained on customer behavioural features. "
            "Churn is defined as no purchase within 180 days after the last transaction. "
            "Leakage-prone recency features were excluded to ensure generalisability."
        ),
        sp(0.5),
    ]
    if ml_metrics:
        story.append(
            tbl(
                [
                    ["Metric",    "Score"],
                    ["Model",     ml_metrics.get("model", "—")],
                    ["ROC AUC",   f"{ml_metrics.get('roc_auc', 0):.4f}"],
                    ["F1 Score",  f"{ml_metrics.get('f1', 0):.4f}"],
                    ["Precision", f"{ml_metrics.get('precision', 0):.4f}"],
                    ["Recall",    f"{ml_metrics.get('recall', 0):.4f}"],
                    ["Accuracy",  f"{ml_metrics.get('accuracy', 0):.4f}"],
                ],
                widths=[8 * cm, 8 * cm],
            )
        )
    story += [
        sp(),
        para(
            f"Current churn rate: <b>{churn:.1f}%</b> of {cust:,} customers "
            "are predicted to have churned based on purchasing behaviour."
        ),
    ]

    # ── 5. AI Executive Summary (optional) ──────────────────────────────────
    if ai_summary and ai_summary.strip():
        story += [PageBreak(), para("5. AI Executive Summary", h1_s), hr()]
        clean = (
            ai_summary
            .replace("**", "")
            .replace("*", "")
            .replace("#", "")
            .strip()
        )
        for blk in clean.split("\n\n"):
            blk = blk.strip()
            if blk:
                story += [para(blk), sp(0.5)]

    doc.build(story)
    return buf.getvalue()
