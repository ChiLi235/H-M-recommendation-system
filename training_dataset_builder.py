"""
training_dataset_builder.py

Builds training datasets for:
  1. Two-Tower retrieval  — positive (user, item) pairs; in-batch negatives during training
  2. SASRec reranker      — positive + hard negative rows with full features

Must be run after feature_engineering.py.

Outputs (./processed/):
  two_tower_train.parquet
  two_tower_val.parquet
  reranker_train.parquet
  reranker_val.parquet
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

INPUT_DIR  = "processed"
OUTPUT_DIR = "processed"

N_NEG_RERANKER = 5    # hard negatives per positive (reranker)
POP_POOL_SIZE  = 200  # top-K popular articles per week used as negative pool
MAX_SEQ_LEN    = 20
RANDOM_SEED    = 42


def _build_week_cutoff_map(year_weeks) -> dict:
    """Pre-compute year_week string → Monday pd.Timestamp for all unique weeks."""
    mapping = {}
    for yw in set(year_weeks):
        yw_clean = yw.replace("-W", "-")
        dt = datetime.strptime(yw_clean + "-1", "%G-%V-%u")
        mapping[yw] = pd.Timestamp(dt)
    return mapping

rng = np.random.default_rng(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    print("Loading data …")
    train_txn  = pd.read_parquet(f"{INPUT_DIR}/transactions_train.parquet")
    val_txn    = pd.read_parquet(f"{INPUT_DIR}/transactions_val.parquet")
    art_feat   = pd.read_parquet(f"{INPUT_DIR}/article_features.parquet")
    art_weekly = pd.read_parquet(f"{INPUT_DIR}/article_weekly_stats.parquet")
    user_seqs  = pd.read_parquet(f"{INPUT_DIR}/user_sequences.parquet")
    uf_train   = pd.read_parquet(f"{INPUT_DIR}/user_features_train.parquet")
    uf_val     = pd.read_parquet(f"{INPUT_DIR}/user_features_val.parquet")

    for df in [train_txn, val_txn]:
        df["date"] = pd.to_datetime(df["date"])

    # Add article_id_enc to weekly stats (art_weekly uses raw article_id)
    id_map = art_feat[["article_id", "article_id_enc"]].drop_duplicates()
    art_weekly = art_weekly.merge(id_map, on="article_id", how="left")

    print(f"  Train txn: {len(train_txn):,}  Val txn: {len(val_txn):,}")
    return train_txn, val_txn, art_feat, art_weekly, user_seqs, uf_train, uf_val


# ══════════════════════════════════════════════════════════════════════════════
# Feature column selection
# ══════════════════════════════════════════════════════════════════════════════

ARTICLE_FEAT_COLS = [
    "article_id_enc",
    "producttype_enc", "indexgroup_enc", "colourgroup_enc", "garmentgroup_enc",
    "article_avg_norm_price", "log_global_sales", "article_channel1_ratio",
]

USER_FEAT_COLS = [
    "customer_id", "customer_id_enc",
    "age", "agebucket_enc", "clubmemberstatus_enc", "fashionnewsfrequency_enc",
    "user_total_purchases", "user_avg_norm_price", "user_purchase_freq",
    "user_recency_days", "user_preferred_channel",
    "user_preferred_prodtype", "user_preferred_indexgroup", "user_preferred_colour",
]


def _sel(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Two-Tower dataset  (positive pairs only)
# ══════════════════════════════════════════════════════════════════════════════

def build_two_tower_dataset(
    txn: pd.DataFrame,
    user_features: pd.DataFrame,
    art_features: pd.DataFrame,
    user_seqs: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    One row per positive purchase.
    User features, article features, and purchase sequences are pre-joined.
    In-batch negatives are generated at training time in the DataLoader.
    """
    print(f"Building two-tower {split_name} …")

    art_sel  = _sel(art_features, ARTICLE_FEAT_COLS)
    user_sel = _sel(user_features, USER_FEAT_COLS)

    df = txn[["customer_id", "article_id", "article_id_enc", "year_week"]].copy()
    df = df.merge(user_sel, on="customer_id", how="inner")
    df = df.merge(art_sel,  on="article_id_enc", how="inner")

    # Join user sequences (only article_id_enc sequence needed for user tower)
    seq_cols = ["customer_id", "seq_article_id_enc", "seq_dates", "seq_len"]
    df = df.merge(user_seqs[seq_cols], on="customer_id", how="left")
    df["seq_len"] = df["seq_len"].fillna(0).astype(int)
    df["seq_article_id_enc"] = df["seq_article_id_enc"].apply(
        lambda x: x if isinstance(x, (list, np.ndarray)) else []
    )
    df["seq_dates"] = df["seq_dates"].apply(
        lambda x: x if isinstance(x, (list, np.ndarray)) else []
    )

    # Temporal filtering: only keep sequence items purchased before the target week
    print(f"  Filtering sequences by temporal cutoff …")
    week_cutoffs = _build_week_cutoff_map(df["year_week"].unique())
    new_seqs = []
    new_lens = []
    for seq_ids, seq_dates, yw in zip(
        df["seq_article_id_enc"], df["seq_dates"], df["year_week"]
    ):
        cutoff = week_cutoffs[yw]
        if len(seq_ids) > 0 and len(seq_dates) > 0:
            filtered = [sid for sid, d in zip(seq_ids, seq_dates) if d < cutoff]
            filtered = filtered[-MAX_SEQ_LEN:]
            new_seqs.append(filtered)
            new_lens.append(len(filtered))
        else:
            new_seqs.append([])
            new_lens.append(0)
    df["seq_article_id_enc"] = new_seqs
    df["seq_len"] = new_lens
    df = df.drop(columns=["seq_dates"])

    # fill NaN in all numeric columns — prevents NaN loss in training
    num_cols = [
        "customer_id_enc", "agebucket_enc", "clubmemberstatus_enc",
        "fashionnewsfrequency_enc", "user_total_purchases", "user_avg_norm_price",
        "user_purchase_freq", "user_recency_days", "user_preferred_channel",
        "user_preferred_prodtype", "user_preferred_indexgroup", "user_preferred_colour",
        "producttype_enc", "indexgroup_enc", "colourgroup_enc", "garmentgroup_enc",
        "article_avg_norm_price", "log_global_sales", "article_channel1_ratio",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"  {len(df):,} positive pairs  ({df['customer_id'].nunique():,} customers)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Reranker dataset  (positives + hard negatives)
# ══════════════════════════════════════════════════════════════════════════════

def _week_popular_pool(txn: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    """Top-pool_size article_id_enc by purchase count per year_week."""
    counts = (
        txn.groupby(["year_week", "article_id_enc"])
        .size()
        .reset_index(name="week_cnt")
    )
    counts["week_rank"] = (
        counts.groupby("year_week")["week_cnt"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return counts[counts["week_rank"] <= pool_size][["year_week", "article_id_enc"]]


def _mine_hard_negatives(
    pos: pd.DataFrame,
    pop_pool: pd.DataFrame,
    n_neg: int,
) -> pd.DataFrame:
    """
    For each (customer_id, year_week) with positives, sample n_neg negatives
    from the weekly popular pool, excluding articles actually purchased.

    Vectorised: one numpy sampling pass per week (40 iterations max),
    no cross-join over millions of customer-week pairs.
    """
    print("  Mining hard negatives (vectorised per-week sampling) …")

    week_pop = pop_pool.groupby("year_week")["article_id_enc"].apply(np.array).to_dict()

    # bought set per (customer, week) for fast exclusion
    pos_set = pos[["customer_id", "article_id_enc", "year_week"]].copy()
    pos_set["_bought"] = 1

    neg_dfs = []
    for week, group in pos.groupby("year_week"):
        pop_arts = week_pop.get(week)
        if pop_arts is None or len(pop_arts) == 0:
            continue

        # unique customers active this week
        custs = group["customer_id"].unique()
        n_cust = len(custs)

        # sample n_neg articles per customer (with replacement from pool)
        sampled_idx  = rng.integers(0, len(pop_arts), size=(n_cust, n_neg))
        sampled_arts = pop_arts[sampled_idx]          # (n_cust, n_neg)

        cust_rep = np.repeat(custs, n_neg)
        art_flat = sampled_arts.flatten()

        neg_df = pd.DataFrame({
            "customer_id":   cust_rep,
            "article_id_enc": art_flat.astype(int),
            "year_week":     week,
            "label":         0,
        })
        neg_dfs.append(neg_df)

    negatives = pd.concat(neg_dfs, ignore_index=True)

    # remove any that are actual positives
    negatives = negatives.merge(
        pos_set, on=["customer_id", "article_id_enc", "year_week"], how="left"
    )
    negatives = (
        negatives[negatives["_bought"].isna()]
        .drop(columns=["_bought"])
        .drop_duplicates(["customer_id", "article_id_enc", "year_week"])
    )

    print(f"  {len(negatives):,} hard negatives")
    return negatives


def build_reranker_dataset(
    txn: pd.DataFrame,
    user_features: pd.DataFrame,
    art_features: pd.DataFrame,
    art_weekly: pd.DataFrame,
    user_seqs: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Builds (customer, candidate_article, label) rows with full features:
      - user demographics + behaviour
      - article metadata + weekly popularity stats
      - user purchase sequence (for SASRec attention)
      - cross features: ever_bought, affinity_prodtype, price_fit
    """
    print(f"Building reranker {split_name} …")

    art_sel  = _sel(art_features, ARTICLE_FEAT_COLS)
    user_sel = _sel(user_features, USER_FEAT_COLS)

    # Positives
    pos = txn[["customer_id", "article_id_enc", "year_week"]].copy()
    pos["label"] = 1
    print(f"  {len(pos):,} positives")

    # Hard negatives
    pop_pool = _week_popular_pool(txn, POP_POOL_SIZE)
    neg = _mine_hard_negatives(pos, pop_pool, N_NEG_RERANKER)

    all_rows = pd.concat([pos, neg], ignore_index=True)

    # ── join features ─────────────────────────────────────────────────────────
    all_rows = all_rows.merge(user_sel, on="customer_id", how="left")
    all_rows = all_rows.merge(art_sel,  on="article_id_enc", how="left")

    # weekly stats
    if art_weekly is not None and "article_id_enc" in art_weekly.columns:
        weekly_sel = art_weekly[
            ["article_id_enc", "year_week", "sales_last_4weeks", "sales_last_8weeks"]
        ].drop_duplicates(subset=["article_id_enc", "year_week"])
        all_rows = all_rows.merge(weekly_sel, on=["article_id_enc", "year_week"], how="left")
        all_rows["sales_last_4weeks"] = all_rows["sales_last_4weeks"].fillna(0).astype(int)
        all_rows["sales_last_8weeks"] = all_rows["sales_last_8weeks"].fillna(0).astype(int)

    # ── cross features (non-temporal) ────────────────────────────────────────
    # user_ever_bought_article is computed inside the temporal loop below
    # to avoid target leakage from txn_history including the target week.

    # user_affinity_prodtype: 1 if article's product type matches user's preferred
    if "producttype_enc" in all_rows.columns and "user_preferred_prodtype" in all_rows.columns:
        all_rows["user_affinity_prodtype"] = (
            all_rows["producttype_enc"] == all_rows["user_preferred_prodtype"]
        ).astype(int)
    else:
        all_rows["user_affinity_prodtype"] = 0

    # user_price_fit: |user avg price - article avg price|
    if "user_avg_norm_price" in all_rows.columns and "article_avg_norm_price" in all_rows.columns:
        all_rows["user_price_fit"] = (
            all_rows["user_avg_norm_price"] - all_rows["article_avg_norm_price"]
        ).abs()
    else:
        all_rows["user_price_fit"] = 0.0

    # ── user purchase sequences & temporal cross features ─────────────────────
    seq_cols = [
        "customer_id",
        "seq_article_id_enc", "seq_producttype_enc", "seq_colourgroup_enc",
        "seq_dates", "seq_len",
    ]
    all_rows = all_rows.merge(user_seqs[seq_cols], on="customer_id", how="left")
    all_rows["seq_len"] = all_rows["seq_len"].fillna(0).astype(int)
    for col in ["seq_article_id_enc", "seq_producttype_enc", "seq_colourgroup_enc", "seq_dates"]:
        all_rows[col] = all_rows[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [])

    # Temporal filtering: only keep sequence items purchased before the target week
    # Also compute user_ever_bought_article from the filtered (pre-cutoff) history
    print(f"  Filtering reranker sequences by temporal cutoff …")
    week_cutoffs = _build_week_cutoff_map(all_rows["year_week"].unique())
    new_arts, new_pts, new_cols, new_lens = [], [], [], []
    ever_bought_list = []

    for s_arts, s_pts, s_cols, s_dates, yw, cand_art in zip(
        all_rows["seq_article_id_enc"], all_rows["seq_producttype_enc"],
        all_rows["seq_colourgroup_enc"], all_rows["seq_dates"],
        all_rows["year_week"], all_rows["article_id_enc"],
    ):
        cutoff = week_cutoffs[yw]
        if len(s_arts) > 0 and len(s_dates) > 0:
            mask = [d < cutoff for d in s_dates]
            arts = [a for a, m in zip(s_arts, mask) if m]

            # temporal ever_bought: check BEFORE slicing to MAX_SEQ_LEN
            ever_bought_list.append(1 if cand_art in arts else 0)

            arts = arts[-MAX_SEQ_LEN:]
            pts  = [p for p, m in zip(s_pts, mask) if m][-MAX_SEQ_LEN:]
            cols = [c for c, m in zip(s_cols, mask) if m][-MAX_SEQ_LEN:]
            new_arts.append(arts)
            new_pts.append(pts)
            new_cols.append(cols)
            new_lens.append(len(arts))
        else:
            new_arts.append([])
            new_pts.append([])
            new_cols.append([])
            new_lens.append(0)
            ever_bought_list.append(0)

    all_rows["seq_article_id_enc"] = new_arts
    all_rows["seq_producttype_enc"] = new_pts
    all_rows["seq_colourgroup_enc"] = new_cols
    all_rows["seq_len"] = new_lens
    all_rows["user_ever_bought_article"] = ever_bought_list
    all_rows = all_rows.drop(columns=["seq_dates"])

    # fill any remaining NaN numerics
    num_cols = [
        "customer_id_enc", "age", "agebucket_enc", "clubmemberstatus_enc",
        "fashionnewsfrequency_enc", "user_total_purchases", "user_avg_norm_price",
        "user_purchase_freq", "user_recency_days", "user_preferred_channel",
        "user_preferred_prodtype", "user_preferred_indexgroup", "user_preferred_colour",
        "producttype_enc", "indexgroup_enc", "colourgroup_enc", "garmentgroup_enc",
        "article_avg_norm_price", "log_global_sales", "article_channel1_ratio",
    ]
    for col in num_cols:
        if col in all_rows.columns:
            all_rows[col] = all_rows[col].fillna(0)

    print(f"  {len(all_rows):,} total reranker rows ({split_name})")
    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    train_txn, val_txn, art_feat, art_weekly, user_seqs, uf_train, uf_val = load_all()

    # ── Two-Tower ─────────────────────────────────────────────────────────────
    tt_train = build_two_tower_dataset(train_txn, uf_train, art_feat, user_seqs, "train")
    tt_train.to_parquet(f"{OUTPUT_DIR}/two_tower_train.parquet", index=False)
    print(f"  Saved two_tower_train.parquet\n")

    tt_val = build_two_tower_dataset(val_txn, uf_val, art_feat, user_seqs, "val")
    tt_val.to_parquet(f"{OUTPUT_DIR}/two_tower_val.parquet", index=False)
    print(f"  Saved two_tower_val.parquet\n")

    # ── Reranker ──────────────────────────────────────────────────────────────
    rer_train = build_reranker_dataset(
        txn=train_txn,
        user_features=uf_train,
        art_features=art_feat,
        art_weekly=art_weekly,
        user_seqs=user_seqs,
        split_name="train",
    )
    rer_train.to_parquet(f"{OUTPUT_DIR}/reranker_train.parquet", index=False)
    print(f"  Saved reranker_train.parquet\n")

    rer_val = build_reranker_dataset(
        txn=val_txn,
        user_features=uf_val,
        art_features=art_feat,
        art_weekly=art_weekly,
        user_seqs=user_seqs,
        split_name="val",
    )
    rer_val.to_parquet(f"{OUTPUT_DIR}/reranker_val.parquet", index=False)
    print(f"  Saved reranker_val.parquet\n")

    print("training_dataset_builder.py complete.")


if __name__ == "__main__":
    main()
