"""
data_preprocessing.py

Loads articles, customers, and transactions; subsamples to last 12 months;
encodes all categoricals; splits into train/val/test; saves processed outputs.

Outputs (written to ./processed/):
  articles_encoded.parquet    -- articles with integer-encoded categoricals
  customers_encoded.parquet   -- customers with integer-encoded features
  transactions_train.parquet  -- 2019-10-01 to 2020-06-30
  transactions_val.parquet    -- 2020-07-01 to 2020-08-31
  transactions_test.parquet   -- 2020-09-01 to 2020-09-22
  encoders.pkl                -- LabelEncoders + vocab sizes for model embedding layers
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── date boundaries ────────────────────────────────────────────────────────────
SUBSAMPLE_START = pd.Timestamp("2019-10-01")   # keep last ~12 months
TRAIN_END       = pd.Timestamp("2020-06-30")
VAL_START       = pd.Timestamp("2020-07-01")
VAL_END         = pd.Timestamp("2020-08-31")
TEST_START      = pd.Timestamp("2020-09-01")

# ── categorical columns to encode ─────────────────────────────────────────────
ARTICLE_CAT_COLS = [
    "product_type_name",
    "product_group_name",
    "index_group_name",
    "department_name",
    "colour_group_name",
    "garment_group_name",
    "index_name",
    "section_name",
]

CUSTOMER_CAT_COLS = [
    "club_member_status",
    "fashion_news_frequency",
]

# ── age bins ───────────────────────────────────────────────────────────────────
AGE_BINS   = [0, 19, 29, 39, 49, 59, 200]
AGE_LABELS = ["teen", "20s", "30s", "40s", "50s", "60+"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. ARTICLES
# ══════════════════════════════════════════════════════════════════════════════

def load_articles() -> pd.DataFrame:
    print("Loading articles.csv …")
    df = pd.read_csv("articles.csv", dtype={"article_id": str})

    # fill rare missing text fields
    for col in ARTICLE_CAT_COLS:
        df[col] = df[col].fillna("Unknown")

    # fill missing price with median
    df["estimated_price_usd"] = df["estimated_price_usd"].fillna(
        df["estimated_price_usd"].median()
    )

    print(f"  {len(df):,} articles, {df['article_id'].nunique():,} unique IDs")
    return df


def encode_articles(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Fit LabelEncoders on article categoricals and add *_enc columns."""
    for col in ARTICLE_CAT_COLS:
        le = LabelEncoder()
        enc_col = col.replace("_name", "").replace("_", "") + "_enc"
        # normalise: lowercase + strip
        series = df[col].str.lower().str.strip()
        df[enc_col] = le.fit_transform(series) + 1  # +1 to reserve 0 for padding
        encoders[enc_col] = {"encoder": le, "vocab_size": len(le.classes_)}
        print(f"  article  {enc_col}: {len(le.classes_)} classes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════

def load_customers() -> pd.DataFrame:
    print("Loading customers.csv …")
    df = pd.read_csv("customers.csv", dtype={"customer_id": str})

    # age: fill missing with median, clip outliers
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age).clip(lower=15, upper=99).astype(int)

    # age bucket
    df["age_bucket"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
    ).astype(str)

    # fill missing categoricals
    df["club_member_status"]    = df["club_member_status"].fillna("UNKNOWN")
    df["fashion_news_frequency"] = df["fashion_news_frequency"].fillna("NONE")
    # normalise "None" vs "NONE" inconsistency observed in the data
    df["fashion_news_frequency"] = (
        df["fashion_news_frequency"].str.strip().str.upper()
    )

    print(f"  {len(df):,} customers")
    return df


def encode_customers(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    for col in CUSTOMER_CAT_COLS + ["age_bucket"]:
        le = LabelEncoder()
        enc_col = col.lower().replace("_", "") + "_enc"
        df[enc_col] = le.fit_transform(df[col].str.lower().str.strip()) + 1  # +1 to reserve 0 for padding
        encoders[enc_col] = {"encoder": le, "vocab_size": len(le.classes_)}
        print(f"  customer {enc_col}: {len(le.classes_)} classes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_transactions() -> pd.DataFrame:
    """
    Load transactions_train_new.csv (custom z-score normalised price).
    Falls back to transactions_train.csv if the new file does not exist.
    """
    path = "transactions_train_new.csv"
    if not os.path.exists(path):
        print("  WARNING: transactions_train_new.csv not found, using transactions_train.csv")
        path = "transactions_train.csv"

    print(f"Loading {path} …")
    df = pd.read_csv(
        path,
        dtype={"customer_id": str, "article_id": str},
        parse_dates=["t_dat"],
    )
    df.rename(columns={"t_dat": "date", "price": "normalized_price"}, inplace=True)
    df["article_id"] = df["article_id"].str.lstrip("0")   # strip leading zeros for join
    print(f"  {len(df):,} rows, date range {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def subsample_and_split(df: pd.DataFrame):
    """Keep last 12 months, then split into train / val / test."""
    df = df[df["date"] >= SUBSAMPLE_START].copy()
    print(f"  After 12-month subsample: {len(df):,} rows")

    train = df[df["date"] <= TRAIN_END]
    val   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
    test  = df[df["date"] >= TEST_START]

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


def add_week_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISO year-week string column (e.g. '2020-W35') for grouping."""
    df = df.copy()
    df["year_week"] = df["date"].dt.strftime("%G-W%V")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. ARTICLE ID ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def align_article_ids(articles: pd.DataFrame, transactions: pd.DataFrame) -> tuple:
    """
    Ensure article_id is consistently a plain integer string without leading zeros.
    Returns updated (articles, transactions).
    """
    articles = articles.copy()
    articles["article_id"] = articles["article_id"].str.lstrip("0")
    transactions = transactions.copy()
    transactions["article_id"] = transactions["article_id"].str.lstrip("0")
    return articles, transactions


def encode_article_ids(articles: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Assign a dense integer index to each article_id (needed for embedding layers)."""
    le = LabelEncoder()
    articles["article_id_enc"] = le.fit_transform(articles["article_id"]) + 1  # +1 to reserve 0 for padding
    encoders["article_id_enc"] = {
        "encoder": le,
        "vocab_size": len(le.classes_),
        # keep raw→enc mapping as dict for fast lookup at inference  (+1 offset applied)
        "id_to_enc": {k: v + 1 for k, v in zip(le.classes_, le.transform(le.classes_))},
    }
    print(f"  article_id_enc: {len(le.classes_):,} unique articles")
    return articles


def encode_customer_ids(customers: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    le = LabelEncoder()
    customers["customer_id_enc"] = le.fit_transform(customers["customer_id"]) + 1  # +1 to reserve 0 for padding
    encoders["customer_id_enc"] = {
        "encoder": le,
        "vocab_size": len(le.classes_),
        "id_to_enc": {k: v + 1 for k, v in zip(le.classes_, le.transform(le.classes_))},
    }
    print(f"  customer_id_enc: {len(le.classes_):,} unique customers")
    return customers


# ══════════════════════════════════════════════════════════════════════════════
# 5. MERGE ENCODED IDS INTO TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def merge_ids_into_transactions(
    txn: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Add article_id_enc and customer_id_enc to transaction rows."""
    art_map  = articles[["article_id",  "article_id_enc"]].drop_duplicates()
    cust_map = customers[["customer_id", "customer_id_enc"]].drop_duplicates()

    txn = txn.merge(art_map,  on="article_id",  how="left")
    txn = txn.merge(cust_map, on="customer_id", how="left")

    # customers/articles that only appear in val/test won't have an enc → -1
    txn["article_id_enc"]  = txn["article_id_enc"].fillna(-1).astype(int)
    txn["customer_id_enc"] = txn["customer_id_enc"].fillna(-1).astype(int)
    return txn


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    encoders = {}

    # ── articles ──────────────────────────────────────────────────────────────
    articles = load_articles()
    articles = encode_articles(articles, encoders)
    articles = encode_article_ids(articles, encoders)

    # keep only columns needed downstream
    article_keep = (
        ["article_id", "article_id_enc", "prod_name",
         "estimated_price_usd", "normalized_price"]
        + [c.replace("_name", "").replace("_", "") + "_enc" for c in ARTICLE_CAT_COLS]
    )
    # normalized_price may not exist in articles.csv — only in transactions_train_new
    article_keep = [c for c in article_keep if c in articles.columns]
    articles_out = articles[article_keep]

    # ── customers ─────────────────────────────────────────────────────────────
    customers = load_customers()
    customers = encode_customers(customers, encoders)
    customers = encode_customer_ids(customers, encoders)

    customer_keep = [
        "customer_id", "customer_id_enc",
        "age", "age_bucket",
        "clubmemberstatus_enc", "fashionnewsfrequency_enc", "agebucket_enc",
    ]
    customers_out = customers[[c for c in customer_keep if c in customers.columns]]

    # ── transactions ──────────────────────────────────────────────────────────
    transactions = load_transactions()
    articles, transactions = align_article_ids(articles, transactions)

    train_txn, val_txn, test_txn = subsample_and_split(transactions)

    for split, df in [("train", train_txn), ("val", val_txn), ("test", test_txn)]:
        df = add_week_column(df)
        df = merge_ids_into_transactions(df, articles_out, customers_out)
        df.to_parquet(f"{OUTPUT_DIR}/transactions_{split}.parquet", index=False)
        print(f"  Saved transactions_{split}.parquet  ({len(df):,} rows)")

    # ── save article + customer tables ────────────────────────────────────────
    articles_out.to_parquet(f"{OUTPUT_DIR}/articles_encoded.parquet", index=False)
    customers_out.to_parquet(f"{OUTPUT_DIR}/customers_encoded.parquet", index=False)
    print(f"  Saved articles_encoded.parquet  ({len(articles_out):,} rows)")
    print(f"  Saved customers_encoded.parquet ({len(customers_out):,} rows)")

    # ── save encoders ─────────────────────────────────────────────────────────
    # also store vocab sizes summary for quick reference when building models
    vocab_sizes = {k: v["vocab_size"] for k, v in encoders.items()}
    encoders["_vocab_sizes"] = vocab_sizes

    with open(f"{OUTPUT_DIR}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    print(f"\nVocab sizes:")
    for k, v in vocab_sizes.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
