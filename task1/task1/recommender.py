"""
Task 1 — Intelligent Reorder Recommendation System
Bristol Regional Food Network Digital Marketplace

Production recommender module. Loads a pre-trained model from disk
rather than retraining on every call.

Usage:
    from recommender import get_recommendations
    recs = get_recommendations(user_id=1, top_n=3)
"""

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "task1_synthetic_orders.csv"
MODEL_PATH = BASE_DIR / "reorder_model.pkl"
ENCODER_PATH = BASE_DIR / "category_encoder.pkl"

# Load model once at import time (not on every call)
_model = None
_encoder = None


def _load_model():
    global _model, _encoder
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run the training notebook first to generate reorder_model.pkl."
            )
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _encoder


def load_order_data() -> pd.DataFrame:
    """Load synthetic purchase history dataset."""
    df = pd.read_csv(DATA_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build user-product feature matrix from purchase history."""
    split_date = df["order_date"].quantile(0.7)
    history_df = df[df["order_date"] <= split_date]
    future_df = df[df["order_date"] > split_date]

    ref_date = history_df["order_date"].max()

    features = (
        history_df.groupby(["user_id", "product_name", "category"])
        .agg(
            frequency=("product_name", "count"),
            total_quantity=("quantity", "sum"),
            avg_quantity=("quantity", "mean"),
            last_purchase_date=("order_date", "max"),
        )
        .reset_index()
    )

    features["days_since_last_purchase"] = (
        ref_date - features["last_purchase_date"]
    ).dt.days

    features["last_purchase_month"] = features["last_purchase_date"].dt.month

    future_pairs = set(zip(future_df["user_id"], future_df["product_name"]))
    features["will_reorder"] = features.apply(
        lambda r: 1 if (r["user_id"], r["product_name"]) in future_pairs else 0,
        axis=1,
    )

    return features


FEATURE_COLS = [
    "frequency",
    "total_quantity",
    "avg_quantity",
    "days_since_last_purchase",
    "last_purchase_month",
    "category_encoded",
]


def get_recommendations(user_id: int, top_n: int = 3) -> list[dict]:
    """
    Return ML-based quick reorder recommendations for a user.

    Args:
        user_id: Customer/user ID.
        top_n:   Number of recommendations to return.

    Returns:
        List of dicts with product details and reorder probability.
    """
    model, encoder = _load_model()

    df = load_order_data()
    features = prepare_features(df)

    features["category_encoded"] = encoder.transform(features["category"])
    X = features[FEATURE_COLS]

    features["reorder_probability"] = model.predict_proba(X)[:, 1]

    user_recs = (
        features[features["user_id"] == int(user_id)]
        .sort_values("reorder_probability", ascending=False)
        .head(top_n)
    )

    recommendations = []
    for _, row in user_recs.iterrows():
        recommendations.append(
            {
                "product_name": row["product_name"],
                "category": row["category"],
                "frequency": int(row["frequency"]),
                "total_quantity": int(row["total_quantity"]),
                "days_since_last_purchase": int(row["days_since_last_purchase"]),
                "reorder_probability": round(float(row["reorder_probability"]) * 100, 2),
                "reason": (
                    f"Recommended because you bought this item "
                    f"{int(row['frequency'])} time(s), with a total of "
                    f"{int(row['total_quantity'])} units ordered, and you last "
                    f"purchased it {int(row['days_since_last_purchase'])} days ago."
                ),
            }
        )

    return recommendations


if __name__ == "__main__":
    # Quick smoke test
    recs = get_recommendations(user_id=1, top_n=3)
    print(f"Recommendations for User 1:")
    for r in recs:
        print(f"  {r['product_name']} ({r['category']}) — {r['reorder_probability']}% probability")
        print(f"    {r['reason']}")
