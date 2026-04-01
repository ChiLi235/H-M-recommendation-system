"""
feature_engineering.py

Computes all features needed by the Two-Tower and SASRec reranker models.
Must be run after data_preprocessing.py.

Outputs (written to ./processed/):
  article_features.parquet        -- global stats + encoded categoricals per article
  article_weekly_stats.parquet    -- rolling 4w/8w sales per (article_id, year_week)
  user_sequences.parquet          -- full sorted purchase sequence per customer
                                     (article_id_enc, producttype_enc, colourgroup_enc)
  user_features_train.parquet     -- user aggregate features using history ≤ 2020-06-30
  user_features_val.parquet       -- user aggregate features using history ≤ 2020-08-31
  user_features_test.parquet      -- user aggregate features using history ≤ 2020-09-22
"""

import os
import pickle
import numpy as np
import pandas as pd

INPUT_DIR  = "processed"
OUTPUT_DIR = "processed"

TRAIN_CUTOFF = pd.Timestamp("2020-06-30")
VAL_CUTOFF   = pd.Timestamp("2020-08-31")
TEST_CUTOFF  = pd.Timestamp("2020-09-22")

MAX_SEQ_LEN = 20   # last-N purchases kept in sequence


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mode_or_default(series: pd.Series, default=-1) -> int:
    """Return the most frequent value in series, or default if empty."""
    if series.empty:
        return default
    return int(series.mode().iloc[0])


def _week_to_monday(year_week: str) -> pd.Timestamp:
    """'2020-W35' → the Monday of that ISO week."""
    return pd.Timestamp.fromisocalendar(
        int(year_week.split("-W")[0]),
        int(year_week.split("-W")[1]),
        1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. ARTICLE GLOBAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_article_global_features(
    train_txn: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join article metadata with global popularity stats derived from training
    transactions only (no val/test leakage).
    """
    print("Computing article global features …")

    stats = (
        train_txn.groupby("article_id")
        .agg(
            article_global_sales   =("customer_id", "count"),
            article_distinct_buyers=("customer_id", "nunique"),
            article_avg_norm_price =("normalized_price", "mean"),
            article_channel1_ratio =("sales_channel_id",
                                     lambda x: (x == 1).mean()),
        )
        .reset_index()
    )

    # join with article metadata
    feat = articles.merge(stats, on="article_id", how="left")
    feat["article_global_sales"]    = feat["article_global_sales"].fillna(0).astype(int)
    feat["article_distinct_buyers"] = feat["article_distinct_buyers"].fillna(0).astype(int)
    feat["article_avg_norm_price"]  = feat["article_avg_norm_price"].fillna(0.0)
    feat["article_channel1_ratio"]  = feat["article_channel1_ratio"].fillna(0.5)
    feat["log_global_sales"]        = np.log1p(feat["article_global_sales"])

    print(f"  {len(feat):,} articles with global features")
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARTICLE WEEKLY ROLLING STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_article_weekly_stats(train_txn: pd.DataFrame) -> pd.DataFrame:
    """
    For every (article_id, year_week) that appears in training, compute:
      - sales_last_4weeks  : purchases in [week-4w, week)
      - sales_last_8weeks  : purchases in [week-8w, week)

    Vectorised implementation:
      1. Build a daily sales pivot  (articles × dates)
      2. Apply a rolling sum (28-day and 56-day windows, closed='left')
      3. Melt back to long format and join to (article, week) pairs
    """
    print("Computing article weekly rolling stats (vectorised) …")

    daily = (
        train_txn.assign(date=pd.to_datetime(train_txn["date"]))
        .groupby(["article_id", "date"])
        .size()
        .reset_index(name="cnt")
    )

    # pivot: rows=date, cols=article_id  (dates as index for rolling)
    pivot = daily.pivot(index="date", columns="article_id", values="cnt").fillna(0)
    pivot = pivot.sort_index()

    # rolling sums — window='28D' / '56D', min_periods=1, closed='left'
    # closed='left' means current day is NOT included → strictly before
    roll4 = pivot.rolling("28D", min_periods=0, closed="left").sum()
    roll8 = pivot.rolling("56D", min_periods=0, closed="left").sum()

    # melt back to long format
    roll4_long = (
        roll4.reset_index()
        .melt(id_vars="date", var_name="article_id", value_name="sales_last_4weeks")
    )
    roll8_long = (
        roll8.reset_index()
        .melt(id_vars="date", var_name="article_id", value_name="sales_last_8weeks")
    )
    daily_rolls = roll4_long.merge(roll8_long, on=["date", "article_id"])

    # map (article, week) → week_start date, then join rolling stats
    weeks = train_txn[["article_id", "year_week"]].drop_duplicates().copy()
    weeks["week_start"] = pd.to_datetime(
        weeks["year_week"].apply(_week_to_monday)
    )

    out = weeks.merge(
        daily_rolls.rename(columns={"date": "week_start"}),
        on=["article_id", "week_start"],
        how="left",
    )
    out["sales_last_4weeks"] = out["sales_last_4weeks"].fillna(0).astype(int)
    out["sales_last_8weeks"] = out["sales_last_8weeks"].fillna(0).astype(int)
    out = out.drop(columns=["week_start"])

    print(f"  {len(out):,} (article, week) stats computed")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. USER PURCHASE SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

def compute_user_sequences(
    train_txn: pd.DataFrame,
    articles: pd.DataFrame,
    max_seq_len: int = MAX_SEQ_LEN,
) -> pd.DataFrame:
    """
    For each customer, build their full chronological purchase sequence using
    training transaction data.  Keeps the last `max_seq_len` purchases.

    Returns one row per customer with list columns:
      seq_article_id_enc    : list[int] length ≤ max_seq_len
      seq_producttype_enc   : list[int]
      seq_colourgroup_enc   : list[int]
      seq_dates             : list[Timestamp]  (for cutoff slicing at training time)
    """
    print("Computing user purchase sequences …")

    # transactions already have article_id_enc from preprocessing;
    # only fetch the columns not yet present
    art_meta = articles[["article_id", "producttype_enc", "colourgroup_enc"]].copy()

    # join article metadata onto transactions
    txn = train_txn.merge(art_meta, on="article_id", how="left")
    txn = txn.sort_values(["customer_id", "date"])

    # fill any unmatched articles (shouldn't happen but be safe)
    for col in ["article_id_enc", "producttype_enc", "colourgroup_enc"]:
        txn[col] = txn[col].fillna(0).astype(int)

    print("  Building per-customer sequences (groupby) …")

    def _build_seq(group: pd.DataFrame) -> dict:
        # take last max_seq_len rows
        g = group.tail(max_seq_len)
        return {
            "seq_article_id_enc":  g["article_id_enc"].tolist(),
            "seq_producttype_enc": g["producttype_enc"].tolist(),
            "seq_colourgroup_enc": g["colourgroup_enc"].tolist(),
            "seq_dates":           g["date"].tolist(),
            "seq_len":             len(g),
        }

    records = []
    for customer_id, group in txn.groupby("customer_id", sort=False):
        seq = _build_seq(group)
        seq["customer_id"] = customer_id
        records.append(seq)

    seqs = pd.DataFrame(records)
    print(f"  {len(seqs):,} customer sequences built")
    return seqs


# ══════════════════════════════════════════════════════════════════════════════
# 4. USER AGGREGATE FEATURES  (computed at a given cutoff date)
# ══════════════════════════════════════════════════════════════════════════════

def compute_user_features(
    txn: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    cutoff: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    """
    Compute one aggregate feature row per customer using transactions
    strictly before `cutoff`.

    Joins with customers table for demographics.
    """
    print(f"Computing user features (cutoff={cutoff.date()}, label={label}) …")

    hist = txn[txn["date"] < cutoff].copy()

    art_meta = articles[["article_id", "producttype_enc",
                          "indexgroup_enc", "colourgroup_enc"]].copy()
    hist = hist.merge(art_meta, on="article_id", how="left")

    # fill missing article metadata
    for col in ["producttype_enc", "indexgroup_enc", "colourgroup_enc"]:
        hist[col] = hist[col].fillna(-1).astype(int)

    print(f"  Aggregating {len(hist):,} transactions …")

    agg = hist.groupby("customer_id").agg(
        user_total_purchases     =("article_id",        "count"),
        user_avg_norm_price      =("normalized_price",  "mean"),
        user_last_purchase_date  =("date",              "max"),
        user_preferred_channel   =("sales_channel_id",  lambda x: _mode_or_default(x)),
        user_preferred_prodtype  =("producttype_enc",   lambda x: _mode_or_default(x)),
        user_preferred_indexgroup=("indexgroup_enc",    lambda x: _mode_or_default(x)),
        user_preferred_colour    =("colourgroup_enc",   lambda x: _mode_or_default(x)),
    ).reset_index()

    # purchase frequency: purchases per week
    # span in weeks from first purchase to cutoff
    first_purchase = hist.groupby("customer_id")["date"].min().rename("first_purchase")
    agg = agg.merge(first_purchase, on="customer_id", how="left")
    span_weeks = ((cutoff - agg["first_purchase"]).dt.days / 7.0).clip(lower=1)
    agg["user_purchase_freq"] = agg["user_total_purchases"] / span_weeks

    # recency: days since last purchase
    agg["user_recency_days"] = (cutoff - agg["user_last_purchase_date"]).dt.days

    # drop helper cols
    agg.drop(columns=["first_purchase", "user_last_purchase_date"], inplace=True)

    # join customer demographics
    agg = agg.merge(
        customers[["customer_id", "customer_id_enc", "age",
                   "clubmemberstatus_enc", "fashionnewsfrequency_enc", "agebucket_enc"]],
        on="customer_id", how="left",
    )

    # customers with no training history still get demographics; fill behaviour cols
    behaviour_cols = [
        "user_total_purchases", "user_avg_norm_price",
        "user_purchase_freq",   "user_recency_days",
        "user_preferred_channel", "user_preferred_prodtype",
        "user_preferred_indexgroup", "user_preferred_colour",
    ]
    for col in behaviour_cols:
        fill = 0 if "preferred" not in col and col != "user_recency_days" else -1
        if col == "user_recency_days":
            fill = 999   # large value → very inactive
        agg[col] = agg[col].fillna(fill)

    print(f"  {len(agg):,} user feature rows")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── load preprocessed files ───────────────────────────────────────────────
    print("Loading preprocessed data …")
    train_txn = pd.read_parquet(f"{INPUT_DIR}/transactions_train.parquet")
    val_txn   = pd.read_parquet(f"{INPUT_DIR}/transactions_val.parquet")
    articles  = pd.read_parquet(f"{INPUT_DIR}/articles_encoded.parquet")
    customers = pd.read_parquet(f"{INPUT_DIR}/customers_encoded.parquet")

    # ensure date column is datetime
    for df in [train_txn, val_txn]:
        df["date"] = pd.to_datetime(df["date"])

    # combined txn up to val cutoff (used for val/test user features)
    trainval_txn = pd.concat([train_txn, val_txn], ignore_index=True)

    # ── article global features ───────────────────────────────────────────────
    art_feat = compute_article_global_features(train_txn, articles)
    art_feat.to_parquet(f"{OUTPUT_DIR}/article_features.parquet", index=False)
    print(f"  Saved article_features.parquet\n")

    # ── article weekly stats ──────────────────────────────────────────────────
    # NOTE: This per-article-per-week computation can be slow on full data.
    # For AWS, it will run on the ml.m5.xlarge preprocessing instance.
    # Locally you can limit to a sample for testing.
    weekly_stats = compute_article_weekly_stats(train_txn)
    weekly_stats.to_parquet(f"{OUTPUT_DIR}/article_weekly_stats.parquet", index=False)
    print(f"  Saved article_weekly_stats.parquet\n")

    # ── user sequences ────────────────────────────────────────────────────────
    # Built from training data only (no leakage into val/test sequences).
    # At training time, training_dataset_builder will slice each user's
    # seq_dates to only include history before the target week.
    user_seqs = compute_user_sequences(train_txn, articles)
    user_seqs.to_parquet(f"{OUTPUT_DIR}/user_sequences.parquet", index=False)
    print(f"  Saved user_sequences.parquet\n")

    # ── user aggregate features at each cutoff ────────────────────────────────
    for label, txn_src, cutoff in [
        ("train",  train_txn,    TRAIN_CUTOFF),
        ("val",    trainval_txn, VAL_CUTOFF),
        ("test",   trainval_txn, TEST_CUTOFF),
    ]:
        uf = compute_user_features(txn_src, articles, customers, cutoff, label)
        uf.to_parquet(f"{OUTPUT_DIR}/user_features_{label}.parquet", index=False)
        print(f"  Saved user_features_{label}.parquet\n")

    print("feature_engineering.py complete.")


if __name__ == "__main__":
    main()
