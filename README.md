# Product Analytics Dashboard

> End-to-end product analytics and machine learning project built on the
> [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
> Designed for portfolio, technical interviews, and internship applications.

---

## Project Overview

This project answers real business questions a Product Manager or Business Analyst
faces every day:

- **Revenue** — Where is money coming from? Is it growing?
- **Customer Behaviour** — Who are our best customers? Who is at risk of churning?
- **Product Performance** — Which products and categories drive the most revenue?
- **Regional Performance** — Which states have the highest order volume?
- **Customer Satisfaction** — How do review scores correlate with churn?
- **Churn Prediction** — Which customers are unlikely to return?

The dashboard provides an interactive view of all of the above, backed by a
trained XGBoost churn model and AI-generated executive summaries via Gemini.

---

## Dataset

| Table | Rows (approx.) | Description |
|---|---|---|
| `customers` | 99k | Customer location and unique ID |
| `orders` | 99k | Core fact table — every purchase event |
| `order_items` | 112k | Line items per order (product, seller, price) |
| `payments` | 103k | Payment method and value per order |
| `reviews` | 99k | Customer satisfaction scores (1–5) |
| `products` | 33k | Product attributes and category |
| `sellers` | 3k | Seller location |
| `geolocation` | 1M | Zip-code latitude/longitude lookup |
| `product_category_translation` | 71 | Portuguese → English category names |

Source: [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data storage | SQLite (stdlib `sqlite3`) |
| Data processing | Pandas, NumPy |
| SQL analysis | 10 production SQL queries |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| AI summaries | Google Gemini API |
| Language | Python 3.10+ |

---

## Folder Structure

```
Product-Analytics-Dashboard/
│
├── data/
│   ├── raw/                         # Original Olist CSVs (git-ignored)
│   ├── customer_features.csv        # Engineered feature table
│   └── customer_features.parquet    # Same — fast I/O for ML
│
├── sql/                             # Schema DDL + 10 analytical SQL queries
│
├── model/
│   └── churn_model.pkl              # Best trained pipeline (git-ignored)
│
├── assets/                          # Plotly HTML charts + SHAP PNG
│
├── database.py         # Downloads CSVs, creates SQLite DB
├── preprocessing.py    # SQL → clean flat DataFrame
├── feature_engineering.py  # Item/order aggregation → customer features
├── eda.py              # EDA charts saved to assets/
├── train.py            # ML pipeline — trains 3 models, saves best
├── predict.py          # Reusable prediction interface for Part 4
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/raghavPahwa27/Product-Analytics-Dashboard.git
cd Product-Analytics-Dashboard

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Kaggle credentials
#    Place kaggle.json in ~/.kaggle/
#    Get yours at: https://www.kaggle.com/settings/account

# 5. Build the database (downloads CSVs + creates SQLite)
python database.py

# 6. Run the dashboard  (Part 3+)
streamlit run app.py
```

---

## Architecture

```
Kaggle API
    │
    ▼
data/raw/  (9 CSV files)
    │
    ▼ database.py
SQLite DB (data/olist.db)
    │
    ├── sql/          (analytical queries — Part 1)
    ├── notebooks/    (EDA — Part 2)
    ├── app.py        (Streamlit dashboard — Part 3)
    ├── model/        (XGBoost churn model — Part 4)
    └── Gemini API    (executive summaries — Part 5)
```

---

## Database Schema

The schema follows a **star schema** pattern with `orders` as the central fact table.

```
customers ──────── orders ─────────── order_items ──── products ── category_translation
                     │                     │
                   payments             sellers
                     │
                   reviews
                     │
                geolocation (via customers/sellers zip prefix)
```

See [`sql/schema.sql`](sql/schema.sql) for the full DDL.

---

## SQL Queries

| File | Business Question |
|---|---|
| `monthly_revenue.sql` | Revenue trend month over month |
| `monthly_orders.sql` | Order volume trend |
| `top_products.sql` | Highest-revenue products |
| `revenue_by_category.sql` | Revenue breakdown by category |
| `orders_by_state.sql` | Regional order and revenue distribution |
| `avg_order_value.sql` | AOV trend over time |
| `repeat_customers.sql` | Customer retention distribution |
| `payment_distribution.sql` | Payment method preferences |
| `avg_review_score.sql` | Satisfaction score by category |
| `top_customers.sql` | Highest lifetime-value customers |

---

## Machine Learning Pipeline (Part 3)

### Algorithms Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model; scaled features, `class_weight="balanced"` |
| Random Forest | Ensemble of 300 trees; handles non-linearity and imbalance natively |
| XGBoost | Gradient-boosted trees; `scale_pos_weight` for imbalance; fastest to converge |

### Evaluation Metrics

| Metric | Why it matters for churn |
|---|---|
| ROC AUC | Primary selection metric — threshold-independent rank quality |
| F1 Score | Harmonic mean of precision / recall — balances both error types |
| Recall | How many churners are correctly caught |
| Precision | How often a churn alert is correct (avoids wasted retention spend) |
| Accuracy | Reported but de-emphasised — misleading on imbalanced targets |

### Model Selection

The best model is selected by **ROC AUC** — the most reliable metric when the
positive class (churned customers) is a minority. AUC measures the model's
ability to rank churned customers above active ones regardless of threshold,
which is what a retention team actually cares about.

### Feature Importance (Top Features)

| Feature | Business meaning |
|---|---|
| `days_since_last_purchase` | Strongest disengagement signal — recency |
| `customer_lifetime_days` | Long-term customers churn far less |
| `total_spend` | High-value customers are retained more aggressively |
| `num_orders` | Repeat buyers have lower churn risk |
| `avg_review_score` | Satisfaction proxy — low scores predict departure |
| `pct_delayed` | Delivery failures drive negative experiences |

### SHAP Explainability

SHAP (SHapley Additive exPlanations) provides per-customer feature attribution
for the best model. The beeswarm plot (`assets/shap_summary.png`) shows how
each feature pushes predictions toward churn or retention for every test-set
customer — making the model auditable and interview-explainable.

### Outputs

```
model/churn_model.pkl          — full sklearn Pipeline (preprocessor + classifier)
assets/roc_curves.html         — ROC curves for all 3 models
assets/confusion_*.html        — confusion matrix per model
assets/feature_importance.html — top-15 feature importance bar chart
assets/shap_summary.png        — SHAP beeswarm summary
```

---

## Future Scope

| Part | Status | Focus |
|---|---|---|
| Part 1 | ✅ Done | Data engineering — SQLite DB + 10 SQL queries |
| Part 2 | ✅ Done | Feature engineering + Exploratory Data Analysis |
| Part 3 | ✅ Done | ML pipeline — LR / RF / XGBoost + SHAP explainability |
| Part 4 | Pending | Interactive Streamlit dashboard with Plotly charts |
| Part 5 | Pending | Gemini AI executive summary and natural-language Q&A |

---

## Author

Built as a portfolio project demonstrating end-to-end skills in data engineering,
SQL, product analytics, machine learning, and dashboard development.
