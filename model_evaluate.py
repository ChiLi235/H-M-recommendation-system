"""
model_evaluate.py

Evaluation metrics for the recommendation pipeline.

Metrics:
  MAP@K    — Mean Average Precision at K  (primary metric, H&M Kaggle standard)
  NDCG@K   — Normalised Discounted Cumulative Gain at K
  HR@K     — Hit Rate at K  (fraction of users with ≥1 hit in top-K)
  Budget utilisation — avg basket_total / budget across test users

Usage:
  python model_evaluate.py
  (or call evaluate_pipeline() from train_pipeline.py / inference.py)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

INPUT_DIR = "processed"
MODEL_DIR = "models"
K = 12


# ══════════════════════════════════════════════════════════════════════════════
# Core metric functions
# ══════════════════════════════════════════════════════════════════════════════

def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """AP@K for a single user."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """NDCG@K for a single user."""
    if not relevant:
        return 0.0
    dcg  = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """HR@K for a single user (1 if any hit, 0 otherwise)."""
    return float(any(item in relevant for item in recommended[:k]))


def compute_metrics(
    recommendations: dict,   # {customer_id: [article_id_enc, ...]} sorted by score
    ground_truth: dict,       # {customer_id: set(article_id_enc)}
    k: int = K,
) -> dict:
    """
    Compute MAP@K, NDCG@K, HR@K over all users in ground_truth.

    Parameters
    ----------
    recommendations : dict
        {customer_id: [article_id_enc, ...]} — ordered list, best first
    ground_truth : dict
        {customer_id: set(article_id_enc)} — items actually purchased
    k : int
        Cutoff for all metrics

    Returns
    -------
    dict with keys map_at_k, ndcg_at_k, hr_at_k, n_users
    """
    ap_scores, ndcg_scores, hr_scores = [], [], []

    for cust_id, relevant in ground_truth.items():
        recs = recommendations.get(cust_id, [])
        ap_scores.append(average_precision_at_k(recs, relevant, k))
        ndcg_scores.append(ndcg_at_k(recs, relevant, k))
        hr_scores.append(hit_rate_at_k(recs, relevant, k))

    return {
        f"map@{k}":  float(np.mean(ap_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"hr@{k}":   float(np.mean(hr_scores)),
        "n_users":   len(ground_truth),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Baseline recommenders
# ══════════════════════════════════════════════════════════════════════════════

def baseline_global_popularity(
    train_txn: pd.DataFrame,
    test_customers: list,
    k: int = K,
    lookback_weeks: int = 4,
) -> dict:
    """Recommend top-K most purchased articles in the last `lookback_weeks` of training."""
    recent = train_txn.sort_values("date")
    cutoff = recent["date"].max() - pd.Timedelta(weeks=lookback_weeks)
    recent = recent[recent["date"] >= cutoff]
    top_k  = (
        recent.groupby("article_id_enc")["customer_id"]
        .count()
        .nlargest(k)
        .index.tolist()
    )
    return {cust: top_k for cust in test_customers}


def baseline_repurchase(
    train_txn: pd.DataFrame,
    test_customers: list,
    k: int = K,
) -> dict:
    """Recommend each user's own most recently purchased articles."""
    recent = (
        train_txn.sort_values("date", ascending=False)
        .groupby("customer_id")["article_id_enc"]
        .apply(lambda x: x.drop_duplicates().head(k).tolist())
        .to_dict()
    )
    global_fallback = (
        train_txn.groupby("article_id_enc")["customer_id"]
        .count().nlargest(k).index.tolist()
    )
    return {
        cust: recent.get(cust, global_fallback)
        for cust in test_customers
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline evaluation
# ══════════════════════════════════════════════════════════════════════════════

def build_ground_truth(test_txn: pd.DataFrame) -> dict:
    """Build {customer_id: set(article_id_enc)} from test transactions."""
    return (
        test_txn.groupby("customer_id")["article_id_enc"]
        .apply(set)
        .to_dict()
    )


def evaluate_pipeline(
    recommendations: dict,
    test_txn: pd.DataFrame,
    k: int = K,
    budget: float = None,
    baskets: dict = None,
    art_prices: dict = None,
) -> dict:
    """
    Full evaluation of the recommendation pipeline.

    Parameters
    ----------
    recommendations : {customer_id: [article_id_enc]}
    test_txn        : test transaction DataFrame
    budget          : if provided, compute budget utilisation
    baskets         : {customer_id: [article_id_enc]} — final basket after knapsack
    art_prices      : {article_id_enc: estimated_price_usd}
    """
    ground_truth = build_ground_truth(test_txn)
    metrics = compute_metrics(recommendations, ground_truth, k=k)

    # Budget utilisation
    if budget is not None and baskets is not None and art_prices is not None:
        util_values = []
        for cust, basket in baskets.items():
            total = sum(art_prices.get(a, 0.0) for a in basket)
            util_values.append(min(total / budget, 1.0))
        metrics["budget_utilisation"] = float(np.mean(util_values))
        metrics["avg_basket_size"]    = float(np.mean([len(b) for b in baskets.values()]))

    return metrics


def print_metrics(metrics: dict, label: str = ""):
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4f}")
        else:
            print(f"  {key:30s}: {val}")
    print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════════════════════════
# Main  (run baselines against test split)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data …")
    train_txn = pd.read_parquet(f"{INPUT_DIR}/transactions_train.parquet")
    test_txn  = pd.read_parquet(f"{INPUT_DIR}/transactions_test.parquet")

    train_txn["date"] = pd.to_datetime(train_txn["date"])
    test_txn["date"]  = pd.to_datetime(test_txn["date"])

    test_customers = test_txn["customer_id"].unique().tolist()
    ground_truth   = build_ground_truth(test_txn)

    print(f"Test users: {len(test_customers):,}  Test transactions: {len(test_txn):,}")

    # ── Baseline 1: global popularity ─────────────────────────────────────────
    pop_recs = baseline_global_popularity(train_txn, test_customers, k=K)
    pop_metrics = compute_metrics(pop_recs, ground_truth, k=K)
    print_metrics(pop_metrics, "Baseline: Global Popularity")

    # ── Baseline 2: repurchase ────────────────────────────────────────────────
    rep_recs = baseline_repurchase(train_txn, test_customers, k=K)
    rep_metrics = compute_metrics(rep_recs, ground_truth, k=K)
    print_metrics(rep_metrics, "Baseline: Repurchase")

    print("\nRun inference.py to evaluate the Two-Tower + SASRec pipeline.")


if __name__ == "__main__":
    main()
