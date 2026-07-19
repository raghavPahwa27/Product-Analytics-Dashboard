"""
predict.py
----------
Reusable prediction interface for the trained churn model.
Designed to be imported by the Streamlit dashboard (Part 4).

Standalone usage:
    python predict.py          -- runs a demo prediction on a sample row

Module usage:
    from predict import predict_churn
    result = predict_churn({"num_orders": 3, "total_spend": 250.0, ...})

Returns
-------
dict with keys:
    predicted_class   : int    0 = Active, 1 = Churned
    churn_probability : float  probability of churn (0.0 - 1.0)
    label             : str    "Active" or "Churned"
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("model/churn_model.pkl")

# Module-level cache: the model is loaded from disk only on the first call.
# Subsequent calls reuse the in-memory pipeline, which matters for Streamlit
# where predict_churn() is called on every user interaction.
_pipeline_cache = None


def load_model(path: Path = MODEL_PATH):
    """Load (and cache) the trained pipeline from disk."""
    global _pipeline_cache
    if _pipeline_cache is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at '{path}'. Run train.py first."
            )
        _pipeline_cache = joblib.load(path)
    return _pipeline_cache


def predict_churn(customer: dict, pipeline=None) -> dict:
    """
    Predict churn for a single customer record.

    Parameters
    ----------
    customer : dict
        Feature values keyed by column name.
        Must include the same columns used during training.
        See train.py -> DROP_COLS and TARGET_COL for exclusions.

    pipeline : fitted sklearn Pipeline, optional
        Pre-loaded pipeline. If None, loads from model/churn_model.pkl.
        Pass an explicit pipeline in tests or when batching predictions
        to avoid repeated disk I/O.

    Returns
    -------
    dict
        predicted_class   : int    0 = Active, 1 = Churned
        churn_probability : float  probability of churn (rounded to 4 dp)
        label             : str    "Active" or "Churned"
    """
    if pipeline is None:
        pipeline = load_model()

    row               = pd.DataFrame([customer])
    predicted_class   = int(pipeline.predict(row)[0])
    churn_probability = float(pipeline.predict_proba(row)[0, 1])

    return {
        "predicted_class":   predicted_class,
        "churn_probability": round(churn_probability, 4),
        "label":             "Churned" if predicted_class == 1 else "Active",
    }


# ---------------------------------------------------------------------------
# Demo (run standalone)
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Sample prediction using representative median-ish values.
    Useful for smoke-testing the saved model after training.
    """
    sample = {
        "total_spend":               250.00,
        "total_products":            3,
        "distinct_categories":       2,
        "distinct_sellers":          2,
        "num_orders":                2,
        "avg_delivery_days":         12.0,
        "avg_review_score":          4.0,
        "pct_delayed":               0.0,
        "weekend_purchase_ratio":    0.5,
        "state":                     "SP",
        "days_since_first_purchase": 200,
        "customer_lifetime_days":    0,
        "preferred_payment_method":  "credit_card",
        "avg_order_value":           125.0,
        "avg_basket_size":           1.5,
        "purchase_frequency":        0.3,
        "avg_freight_ratio":         0.15,
    }

    result = predict_churn(sample)

    print("\nDemo prediction")
    print("=" * 40)
    for key, value in result.items():
        print(f"  {key:22s}: {value}")
    print("=" * 40)
    print()


if __name__ == "__main__":
    _demo()
