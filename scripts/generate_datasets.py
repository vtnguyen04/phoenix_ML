"""
Generate Datasets — Download/create all training datasets to data/.

Downloads real datasets and saves them as CSV/NPZ files that are:
    1. Loaded by DataLoaders during training and retraining
    2. Stored in data/ directory (gitignored)

Usage:
    uv run python scripts/generate_datasets.py            # All datasets
    uv run python scripts/generate_datasets.py --model credit-risk  # One dataset

This is idempotent — re-running will overwrite existing files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def generate_credit_risk() -> None:
    """Download German Credit dataset from OpenML → CSV."""
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    print("📥 [credit-risk] Fetching German Credit from OpenML...")
    os.makedirs(os.path.expanduser("~/scikit_learn_data"), exist_ok=True)
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame

    numeric = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    categorical = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ]
    )

    x_base = preprocessor.fit_transform(df[numeric + categorical]).astype(np.float32)

    # Feature engineering (10 extra features)
    eps = 1e-8
    dur = x_base[:, 0:1]
    credit = x_base[:, 1:2]
    inst = x_base[:, 2:3]
    age = x_base[:, 4:5]
    existing = x_base[:, 5:6]
    emp_i = len(numeric) + categorical.index("employment")
    chk_i = len(numeric) + categorical.index("checking_status")
    sav_i = len(numeric) + categorical.index("savings_status")
    emp = x_base[:, emp_i : emp_i + 1]
    chk = x_base[:, chk_i : chk_i + 1]
    sav = x_base[:, sav_i : sav_i + 1]

    engineered = np.hstack(
        [
            credit / (dur + eps),
            age / (credit + eps),
            inst / (credit + eps),
            age * (emp + 1),
            credit * inst / (dur + eps),
            dur * inst,
            chk * sav,
            age * chk,
            credit * existing,
            np.log1p(np.abs(credit)),
        ]
    )

    all_features = (
        numeric
        + categorical
        + [
            "credit_per_month",
            "age_credit_ratio",
            "installment_credit_ratio",
            "age_employment_score",
            "credit_risk_density",
            "duration_installment",
            "checking_savings_interact",
            "age_checking_interact",
            "credit_existing_interact",
            "log_credit_amount",
        ]
    )

    x_all = np.hstack([x_base, engineered]).astype(np.float32)
    y_all = (df["class"] == "good").astype(int).values

    out_dir = DATA_DIR / "credit_risk"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(x_all, columns=all_features)
    out_df["target"] = y_all
    out_df.to_csv(out_dir / "dataset.csv", index=False)
    print(f"   ✅ {len(out_df)} samples × {len(all_features)} features → {out_dir / 'dataset.csv'}")


def generate_fraud_detection() -> None:
    """Generate reproducible synthetic fraud dataset → CSV."""
    from sklearn.datasets import make_classification

    print("📥 [fraud-detection] Generating synthetic fraud dataset...")

    feature_names = [
        "transaction_amount",
        "merchant_category",
        "card_type",
        "transaction_hour",
        "distance_from_home",
        "distance_from_last_txn",
        "ratio_to_median_amount",
        "is_weekend",
        "is_night",
        "num_txn_last_24h",
        "num_txn_last_7d",
        "avg_amount_last_30d",
    ]

    x_data, y_data = make_classification(
        n_samples=5000,
        n_features=len(feature_names),
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=42,
    )
    x_data = x_data.astype(np.float32)

    out_dir = DATA_DIR / "fraud_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(x_data, columns=feature_names)
    out_df["target"] = y_data
    out_df.to_csv(out_dir / "dataset.csv", index=False)
    path = out_dir / "dataset.csv"
    print(f"   ✅ {len(out_df)} samples × {len(feature_names)} features → {path}")


def generate_house_price() -> None:
    """Download California Housing dataset → CSV."""
    from sklearn.datasets import fetch_california_housing

    print("📥 [house-price] Fetching California Housing...")
    data = fetch_california_housing()
    feature_names = list(data.feature_names)

    out_dir = DATA_DIR / "house_price"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(data.data.astype(np.float32), columns=feature_names)
    out_df["target"] = data.target.astype(np.float32)
    out_df.to_csv(out_dir / "dataset.csv", index=False)
    csv_path = out_dir / "dataset.csv"
    print(f"   ✅ {len(out_df)} samples × {len(feature_names)} features → {csv_path}")


def generate_image_classification() -> None:
    """Download Fashion-MNIST → NPZ."""
    from sklearn.datasets import fetch_openml

    print("📥 [image-class] Fetching Fashion-MNIST from OpenML...")
    os.makedirs(os.path.expanduser("~/scikit_learn_data"), exist_ok=True)
    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")

    x_all = data.data.astype(np.float32)
    y_all = data.target.astype(int)

    # Use a manageable subset (25k) for faster training
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(x_all))[:25000]
    x_subset = x_all[indices]
    y_subset = y_all[indices]

    out_dir = DATA_DIR / "image_class"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "dataset.npz",
        X=x_subset,
        y=y_subset,
    )
    print(f"   ✅ {len(x_subset)} images × 784 pixels → {out_dir / 'dataset.npz'}")

    # Save class names metadata
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"class_names": class_names, "n_classes": 10, "image_size": "28x28"}, f, indent=2)


def generate_sentiment() -> None:
    """Download real IMDB movie reviews dataset → CSV.

    Downloads the IMDB Large Movie Review Dataset (50K reviews)
    from Stanford AI Lab. This is a standard NLP benchmark with
    real movie review text and binary sentiment labels.

    Source: https://ai.stanford.edu/~amaas/data/sentiment/
    """
    import tarfile
    import urllib.request

    print("📥 [sentiment] Downloading IMDB movie reviews (real dataset)...")

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = DATA_DIR / "aclImdb_v1.tar.gz"
    extract_dir = DATA_DIR / "aclImdb"

    # Download if not cached
    if not tar_path.exists():
        print(f"   Downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"   Downloaded: {tar_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"   Using cached: {tar_path}")

    # Extract if not already extracted
    if not extract_dir.exists():
        print("   Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)

    # Read reviews from directory structure:
    #   aclImdb/train/pos/*.txt  → positive reviews
    #   aclImdb/train/neg/*.txt  → negative reviews
    #   aclImdb/test/pos/*.txt   → positive reviews
    #   aclImdb/test/neg/*.txt   → negative reviews
    reviews = []
    labels = []

    for split in ["train", "test"]:
        for label_name, label_val in [("pos", 1), ("neg", 0)]:
            review_dir = extract_dir / split / label_name
            if not review_dir.exists():
                print(f"   ⚠️ Missing: {review_dir}")
                continue
            for txt_file in sorted(review_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8").strip()
                # Clean HTML tags
                text = text.replace("<br />", " ").replace("<br/>", " ")
                reviews.append(text)
                labels.append(label_val)

    if not reviews:
        print("   ❌ No reviews found. Falling back to synthetic data.")
        return

    out_dir = DATA_DIR / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame({"review": reviews, "sentiment": labels})
    # Shuffle with fixed seed for reproducibility
    out_df = out_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_df.to_csv(out_dir / "dataset.csv", index=False)

    n_pos = out_df["sentiment"].sum()
    n_neg = len(out_df) - n_pos
    print(f"   ✅ {len(out_df)} real reviews → {out_dir / 'dataset.csv'}")
    print(f"      Positive: {n_pos}, Negative: {n_neg}")


# ─── Registry ───────────────────────────────────────────────────

GENERATORS: dict[str, tuple[str, object]] = {
    "credit-risk": ("Credit Risk (German Credit)", generate_credit_risk),
    "fraud-detection": ("Fraud Detection (Synthetic)", generate_fraud_detection),
    "house-price": ("House Price (California Housing)", generate_house_price),
    "image-class": ("Image Classification (Fashion-MNIST)", generate_image_classification),
    "sentiment": ("Sentiment Analysis (Synthetic)", generate_sentiment),
}


def main() -> None:
    """Generate datasets for all or specific models."""
    parser = argparse.ArgumentParser(description="Generate training datasets")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(GENERATORS.keys()),
        help="Generate dataset for specific model. Default: all.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🗂️  Data directory: {DATA_DIR}\n")

    targets = [args.model] if args.model else list(GENERATORS.keys())

    for model_id in targets:
        name, gen_fn = GENERATORS[model_id]
        print(f"━━━ {name} ━━━")
        try:
            gen_fn()  # type: ignore[operator]
        except Exception as e:
            print(f"   ❌ Failed: {e}", file=sys.stderr)
        print()

    print("✅ Done! Train models with:")
    print("   uv run python examples/credit_risk/train.py")
    print("   uv run python examples/house_price/train.py")
    print("   uv run python examples/sentiment/train.py")


if __name__ == "__main__":
    main()
