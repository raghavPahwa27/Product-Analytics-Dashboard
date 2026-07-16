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
│   ├── raw/            # Original Olist CSVs (git-ignored)
│   └── processed/      # Feature-engineered datasets (git-ignored)
│
├── sql/                # Schema DDL + 10 analytical SQL queries
│
├── models/             # Trained ML model artefacts (git-ignored)
│
├── database.py         # Downloads CSVs, creates SQLite DB
├── requirements.txt    # Python dependencies
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

## Future Scope

| Part | Focus |
|---|---|
| Part 2 | Exploratory Data Analysis — Jupyter notebooks, statistical summaries |
| Part 3 | Interactive Streamlit dashboard with Plotly charts |
| Part 4 | XGBoost churn prediction model + SHAP explainability |
| Part 5 | Gemini AI executive summary and natural-language Q&A |

---

## Author

Built as a portfolio project demonstrating end-to-end skills in data engineering,
SQL, product analytics, machine learning, and dashboard development.
