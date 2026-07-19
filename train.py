"""
train.py
--------
Part 3: End-to-end ML pipeline for customer churn prediction.

Models compared : Logistic Regression, Random Forest, XGBoost
Best model saved: model/churn_model.pkl  (full sklearn Pipeline)

Plots saved to assets/
    roc_curves.html
    confusion_logistic_regression.html
    confusion_random_forest.html
    confusion_xgboost.html
    feature_importance.html
    shap_summary.png

Run:
    python train.py
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_PARQUET = Path("data/customer_features.parquet")
FEATURES_CSV     = Path("data/customer_features.csv")
MODEL_DIR        = Path("model")
ASSETS_DIR       = Path("assets")
RANDOM_STATE     = 42

# Leakage exclusions — important for correctness and interview credibility:
#
# days_since_last_purchase  — directly defines churn (> 180 days = churned).
# days_since_first_purchase — 93% of customers have exactly 1 order, so for
#   them first_purchase == last_purchase and this column equals the direct leak.
# purchase_frequency        — computed as num_orders / days_since_first_purchase,
#   so it implicitly encodes the same information for single-order customers.
#
# Dropping all three forces the model to learn from genuine behavioural signals
# (spend, reviews, delivery quality, category diversity) and produces a
# realistic, defensible AUC rather than a trivially perfect one.
DROP_COLS = [
    "customer_unique_id",
    "city",
    "days_since_last_purchase",
    "days_since_first_purchase",
    "purchase_frequency",
]

CATEGORICAL_COLS = ["state", "preferred_payment_method"]
TARGET_COL       = "churn"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Parquet preferred; CSV as fallback."""
    if FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
        log.info("Loaded parquet  shape=%s", df.shape)
    elif FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
        log.info("Loaded CSV (fallback)  shape=%s", df.shape)
    else:
        raise FileNotFoundError(
            "Run feature_engineering.py first to generate customer_features.parquet."
        )
    return df


# ---------------------------------------------------------------------------
# 2. Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Drop non-feature columns; split X and y.

    Returns
    -------
    X         : pd.DataFrame   raw features (categoricals still strings)
    y         : pd.Series      binary churn label
    num_cols  : list[str]      numeric column names
    cat_cols  : list[str]      categorical column names
    """
    df       = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X        = df.drop(columns=[TARGET_COL])
    y        = df[TARGET_COL].astype(int)
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    log.info("Numeric features    : %d", len(num_cols))
    log.info("Categorical features: %d  %s", len(cat_cols), cat_cols)
    log.info("Churn rate          : %.1f%%", 100 * y.mean())
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Numeric  -> median imputation
      Median is robust to the right-skewed spend / delivery-days distributions.

    Categorical -> most-frequent imputation + OrdinalEncoder
      OrdinalEncoder is used for all three models for simplicity.
      Tree models handle ordinal-encoded categoricals natively.
      For Logistic Regression this is a mild approximation; OneHotEncoder
      would be strictly correct but adds columns without meaningfully
      improving a portfolio-level result.
    """
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """80/20 stratified split; stratify ensures both splits mirror churn rate."""
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# 4. Model pipelines
# ---------------------------------------------------------------------------

def build_pipelines(preprocessor: ColumnTransformer, scale_pos_weight: float) -> dict:
    """
    One sklearn Pipeline per model.

    Each pipeline clones the preprocessor so every model owns its own
    fitted transformer -- no shared state, no data leakage.

    Hyperparameter choices (intentional defaults, no GridSearchCV):
      class_weight="balanced"  -- LR and RF compensate for churn imbalance.
      scale_pos_weight         -- XGBoost equivalent of class_weight="balanced".
      n_estimators=300         -- enough to stabilise variance without long waits.
      StandardScaler for LR    -- gradient descent converges faster on scaled data.
    """
    return {
        "Logistic Regression": Pipeline([
            ("pre",    clone(preprocessor)),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1_000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre", clone(preprocessor)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("pre", clone(preprocessor)),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# 5. Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    pipelines: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list:
    """Fit each pipeline and collect evaluation metrics."""
    results = []
    for name, pipeline in pipelines.items():
        log.info("Training %s ...", name)
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
            "ROC AUC":   roc_auc_score(y_test, y_proba),
            # Private keys -- used by plot helpers, excluded from display table
            "_pipeline": pipeline,
            "_y_pred":   y_pred,
            "_y_proba":  y_proba,
        })
        log.info("  AUC=%.4f  F1=%.4f", results[-1]["ROC AUC"], results[-1]["F1"])
    return results


def print_comparison(results: list) -> None:
    """
    Print a formatted comparison table sorted by ROC AUC.

    ROC AUC is the primary selection metric -- it measures rank separation
    between classes regardless of threshold, making it more reliable than
    accuracy when the positive class (churned) is a minority.
    """
    metric_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    df = (
        pd.DataFrame(results)[metric_cols]
        .round(4)
        .sort_values("ROC AUC", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

    best = df.iloc[0]
    print(f"\n  Best model : {best['Model']}")
    print(f"  ROC AUC   : {best['ROC AUC']:.4f}")
    print(f"  F1 Score  : {best['F1']:.4f}")
    print(
        "\n  Selection criterion: ROC AUC\n"
        "  Churn labels are imbalanced. AUC measures rank separation\n"
        "  regardless of threshold, which is more reliable than accuracy\n"
        "  on an imbalanced target.\n"
    )


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(results: list, y_test: pd.Series) -> None:
    """Multi-line ROC chart -- all models on one canvas for easy comparison."""
    fig = go.Figure()
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{r['Model']}  (AUC={r['ROC AUC']:.3f})",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="grey"),
        name="Random baseline",
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05),
    )
    fig.write_html(ASSETS_DIR / "roc_curves.html")
    log.info("Saved: assets/roc_curves.html")


def plot_confusion_matrix(name: str, y_test: pd.Series, y_pred: np.ndarray) -> None:
    cm     = confusion_matrix(y_test, y_pred)
    labels = ["Active (0)", "Churned (1)"]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        title=f"Confusion Matrix — {name}",
    )
    slug = name.lower().replace(" ", "_")
    fig.write_html(ASSETS_DIR / f"confusion_{slug}.html")
    log.info("Saved: assets/confusion_%s.html", slug)


def _feature_names(preprocessor: ColumnTransformer) -> list:
    """Strip 'num__' / 'cat__' prefixes from ColumnTransformer output names."""
    return [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]


def plot_feature_importance(pipeline: Pipeline, top_n: int = 15) -> None:
    """
    Horizontal bar chart of the top-N feature importances.

    Uses the classifier's built-in feature_importances_ (RF / XGBoost).
    Skipped for Logistic Regression -- coef_ magnitudes are not directly
    comparable to tree-based importances and require additional normalisation.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        log.info("Skipping feature importance -- not available for %s", type(clf).__name__)
        return

    names  = _feature_names(pipeline.named_steps["pre"])
    imp    = pd.Series(clf.feature_importances_, index=names).nlargest(top_n).sort_values()
    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]

    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances — {type(clf).__name__}",
        labels={"feature": "Feature", "importance": "Importance Score"},
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    fig.write_html(ASSETS_DIR / "feature_importance.html")
    log.info("Saved: assets/feature_importance.html")

    print("\nTop 5 features and their business meaning:")
    business = {
        "days_since_last_purchase":  "How recently the customer bought — the strongest signal of disengagement",
        "customer_lifetime_days":    "Span of purchase history — long-term customers churn less",
        "days_since_first_purchase": "How long the customer has been in the dataset",
        "total_spend":               "Overall spend — high-value customers are retained more aggressively",
        "num_orders":                "Order count — repeat buyers are less likely to churn",
        "avg_review_score":          "Satisfaction proxy — low scores correlate with disengagement",
        "pct_delayed":               "Delivery reliability — delays drive negative experiences",
        "purchase_frequency":        "Order cadence — infrequent buyers are higher churn risk",
    }
    for feat in imp.tail(5).index[::-1]:
        meaning = business.get(feat, "—")
        print(f"  {feat:35s}  {meaning}")
    print()


def plot_shap(pipeline: Pipeline, X_test: pd.DataFrame, sample_size: int = 500) -> None:
    """
    SHAP summary (beeswarm) for the best model.

    TreeExplainer is used for RF / XGBoost -- it is exact and fast.
    LinearExplainer is used for Logistic Regression.

    A random sub-sample of `sample_size` rows keeps rendering fast
    without losing the overall distribution shape.
    Saved as PNG -- matplotlib is the native SHAP rendering backend.
    """
    clf       = pipeline.named_steps["clf"]
    pre_steps = pipeline.steps[:-1]     # every step except the final classifier
    X_trans   = Pipeline(pre_steps).transform(X_test)
    names     = _feature_names(pipeline.named_steps["pre"])

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_trans), size=min(sample_size, len(X_trans)), replace=False)
    X_s = X_trans[idx]
    # np.array required: shap.summary_plot indexes feature_names with a numpy
    # integer array (sort_inds), which fails on a plain Python list.
    names_arr = np.array(names)

    if isinstance(clf, (XGBClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_s)
        # RandomForestClassifier returns [class0_vals, class1_vals] -- take class 1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    else:
        explainer = shap.LinearExplainer(clf, X_s)
        shap_vals = explainer.shap_values(X_s)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_s, feature_names=names_arr, show=False)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: assets/shap_summary.png")


# ---------------------------------------------------------------------------
# 7. Model selection & persistence
# ---------------------------------------------------------------------------

def select_best(results: list) -> dict:
    """Return the result entry with the highest ROC AUC."""
    return max(results, key=lambda r: r["ROC AUC"])


def save_model(pipeline: Pipeline) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "churn_model.pkl")
    log.info("Saved: model/churn_model.pkl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)

    # 1. Load
    df = load_data()

    # 2. Prepare
    X, y, num_cols, cat_cols = prepare_features(df)

    # 3. Split (stratified 80/20)
    X_train, X_test, y_train, y_test = split_data(X, y)
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    # 4. Build pipelines
    # scale_pos_weight = neg / pos tells XGBoost how much more to penalise
    # missing a churned customer relative to missing an active one.
    neg, pos         = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info("scale_pos_weight=%.2f  (neg=%d / pos=%d)", scale_pos_weight, neg, pos)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipelines    = build_pipelines(preprocessor, scale_pos_weight)

    # 5. Train & evaluate all models
    results = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)
    print_comparison(results)

    # 6. Plots for every model
    plot_roc_curves(results, y_test)
    for r in results:
        plot_confusion_matrix(r["Model"], y_test, r["_y_pred"])

    # 7. Deeper analysis for the best model only
    best      = select_best(results)
    best_pipe = best["_pipeline"]
    log.info("Best model: %s  (AUC=%.4f)", best["Model"], best["ROC AUC"])

    plot_feature_importance(best_pipe)
    plot_shap(best_pipe, X_test)

    # 8. Save
    save_model(best_pipe)
    print(f"\n  Pipeline saved -> model/churn_model.pkl")
    print(f"  Plots saved    -> assets/")
    print(f"\n  Run  python predict.py  to test a sample prediction.")


if __name__ == "__main__":
    run()
