"""
inference.py

End-to-end inference pipeline:
  1. Two-Tower retrieval  → top-500 candidates per user (FAISS ANN)
  2. SASRec reranker      → score each candidate
  3. Greedy knapsack      → select budget-feasible basket

Usage:
  python inference.py --budget 100 --customer_id <id>

  Or import recommend() for programmatic use.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIR      = os.environ.get("SM_CHANNEL_PROCESSED", "processed")
MODEL_DIR      = os.environ.get("SM_CHANNEL_MODELS", "models")
CHECKPOINT_DIR = "/opt/ml/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RETRIEVAL_TOP_K = 500   # Two-Tower retrieves this many candidates
RERANK_BATCH    = 512   # batch size for reranker scoring
MAX_SEQ_LEN_TT  = 20   # Two-Tower sequence length
MAX_SEQ_LEN_RR  = 10   # Reranker sequence length (must match reranker_model.py MAX_SEQ_LEN)
MAX_SEQ_LEN     = MAX_SEQ_LEN_TT  # default kept for backward compat


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_two_tower(vocab_sizes: dict):
    from two_tower_model import TwoTowerModel
    model = TwoTowerModel(vocab_sizes)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "two_tower_final.pt"), map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    return model


def load_reranker(vocab_sizes: dict):
    from reranker_model import SASRecReranker
    model = SASRecReranker(vocab_sizes)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "reranker_final.pt"), map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    return model


def load_faiss_index():
    import faiss
    index      = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))
    art_ids    = np.load(os.path.join(MODEL_DIR, "faiss_article_ids.npy"))
    return index, art_ids


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Two-Tower retrieval
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_batch(
    customer_row: pd.Series,
    user_seq: pd.Series,
    vocab_sizes: dict,
) -> dict:
    """Build a single-sample user input batch for the Two-Tower user tower."""
    def _pad_seq(lst):
        lst = list(lst)[-MAX_SEQ_LEN_TT:] if len(lst) > MAX_SEQ_LEN_TT else list(lst)
        ids  = torch.zeros(MAX_SEQ_LEN_TT, dtype=torch.long)
        mask = torch.zeros(MAX_SEQ_LEN_TT, dtype=torch.bool)
        if lst:
            ids[:len(lst)]  = torch.tensor(lst, dtype=torch.long)
            mask[:len(lst)] = True
        return ids, mask

    seq_ids, seq_mask = _pad_seq(
        user_seq["seq_article_id_enc"] if user_seq is not None else []
    )
    seq_ids = seq_ids.clamp(0, vocab_sizes["article_id_enc"])

    art_max  = vocab_sizes["article_id_enc"]
    age_max  = vocab_sizes["agebucket_enc"]
    club_max = vocab_sizes["clubmemberstatus_enc"]
    news_max = vocab_sizes["fashionnewsfrequency_enc"]

    def _safe_int(row, key, default, max_val):
        v = row.get(key, default)
        return min(int(v) if (v is not None and v == v) else default, max_val)  # v==v guards NaN

    def _safe_float(row, key, default):
        v = row.get(key, default)
        return float(v) if (v is not None and v == v) else default

    return {
        "age_enc":  torch.tensor([_safe_int(customer_row, "agebucket_enc",           0, age_max)],  dtype=torch.long),
        "club_enc": torch.tensor([_safe_int(customer_row, "clubmemberstatus_enc",     0, club_max)], dtype=torch.long),
        "news_enc": torch.tensor([_safe_int(customer_row, "fashionnewsfrequency_enc", 0, news_max)], dtype=torch.long),
        "user_num": torch.tensor([[
            _safe_float(customer_row, "user_total_purchases",  0.0),
            _safe_float(customer_row, "user_avg_norm_price",   0.0),
            _safe_float(customer_row, "user_purchase_freq",    0.0),
            _safe_float(customer_row, "user_recency_days",   999.0),
            _safe_float(customer_row, "user_preferred_channel", 0.0),
        ]], dtype=torch.float32),
        "seq_ids":  seq_ids.unsqueeze(0),   # (1, L)
        "seq_mask": seq_mask.unsqueeze(0),  # (1, L)
    }


def retrieve_candidates(
    two_tower_model,
    faiss_index,
    faiss_art_ids: np.ndarray,
    customer_row: pd.Series,
    user_seq,
    vocab_sizes: dict,
    top_k: int = RETRIEVAL_TOP_K,
) -> np.ndarray:
    """
    Compute user embedding → FAISS ANN search → return top_k article_id_enc array.
    """
    user_batch = {k: v.to(DEVICE) for k, v in
                  _build_user_batch(customer_row, user_seq, vocab_sizes).items()}

    with torch.no_grad():
        user_emb = two_tower_model.user_embed(user_batch).cpu().numpy().astype(np.float32)

    _, indices = faiss_index.search(user_emb, top_k)
    candidate_article_encs = faiss_art_ids[indices[0]]
    return candidate_article_encs


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: SASRec reranking
# ══════════════════════════════════════════════════════════════════════════════

def _build_reranker_batch(
    customer_row: pd.Series,
    user_seq,
    candidates: np.ndarray,
    art_features: pd.DataFrame,
    art_weekly: pd.DataFrame,
    year_week: str,
    vocab_sizes: dict = None,
) -> dict:
    """Build a batch of (candidate × features) tensors for the reranker."""
    n = len(candidates)

    # Sequence (same for all candidates)
    def _pad_seq(lst, maxlen):
        lst = list(lst)[-maxlen:] if len(lst) > maxlen else list(lst)
        t = torch.zeros(maxlen, dtype=torch.long)
        if lst:
            t[:len(lst)] = torch.tensor(lst, dtype=torch.long)
        return t

    seq_ids = _pad_seq(user_seq["seq_article_id_enc"]  if user_seq is not None else [], MAX_SEQ_LEN_RR)
    seq_pt  = _pad_seq(user_seq["seq_producttype_enc"] if user_seq is not None else [], MAX_SEQ_LEN_RR)
    seq_col = _pad_seq(user_seq["seq_colourgroup_enc"] if user_seq is not None else [], MAX_SEQ_LEN_RR)
    seq_len = int(user_seq["seq_len"]) if user_seq is not None else 0
    seq_mask = torch.zeros(MAX_SEQ_LEN_RR, dtype=torch.bool)
    seq_mask[:min(seq_len, MAX_SEQ_LEN_RR)] = True

    # Clamp sequence IDs to valid embedding range
    if vocab_sizes is not None:
        seq_ids = seq_ids.clamp(0, vocab_sizes["article_id_enc"])
        seq_pt  = seq_pt.clamp(0,  vocab_sizes["producttype_enc"])
        seq_col = seq_col.clamp(0, vocab_sizes["colourgroup_enc"])

    # Expand sequence to (n, L)
    seq_ids_b  = seq_ids.unsqueeze(0).expand(n, -1)
    seq_pt_b   = seq_pt.unsqueeze(0).expand(n, -1)
    seq_col_b  = seq_col.unsqueeze(0).expand(n, -1)
    seq_mask_b = seq_mask.unsqueeze(0).expand(n, -1)

    # Article features for each candidate
    art_sel = art_features.set_index("article_id_enc")

    def _get_art(enc, col, default=0):
        try:
            return float(art_sel.at[enc, col])
        except (KeyError, TypeError):
            return float(default)

    pt_encs  = [min(int(_get_art(a, "producttype_enc")),  vocab_sizes["producttype_enc"]  if vocab_sizes else 9999) for a in candidates]
    ig_encs  = [min(int(_get_art(a, "indexgroup_enc")),   vocab_sizes["indexgroup_enc"]   if vocab_sizes else 9999) for a in candidates]
    cg_encs  = [min(int(_get_art(a, "colourgroup_enc")),  vocab_sizes["colourgroup_enc"]  if vocab_sizes else 9999) for a in candidates]
    gg_encs  = [min(int(_get_art(a, "garmentgroup_enc")), vocab_sizes["garmentgroup_enc"] if vocab_sizes else 9999) for a in candidates]
    avg_prices = [_get_art(a, "article_avg_norm_price") for a in candidates]
    log_sales  = [_get_art(a, "log_global_sales")       for a in candidates]
    ch1_ratio  = [_get_art(a, "article_channel1_ratio") for a in candidates]

    # Weekly stats
    if art_weekly is not None and year_week is not None:
        wk = art_weekly[art_weekly["year_week"] == year_week].set_index("article_id_enc")
        s4 = [float(wk.at[a, "sales_last_4weeks"])  if a in wk.index else 0.0 for a in candidates]
        s8 = [float(wk.at[a, "sales_last_8weeks"])  if a in wk.index else 0.0 for a in candidates]
    else:
        s4, s8 = [0.0] * n, [0.0] * n

    item_cat = torch.tensor(list(zip(pt_encs, ig_encs, cg_encs, gg_encs)), dtype=torch.long)
    item_num = torch.tensor(list(zip(avg_prices, log_sales, ch1_ratio, s4, s8)), dtype=torch.float32)

    # User features (same for all candidates) — NaN-safe for cold-start users
    def _si(key, default, max_val):
        v = customer_row.get(key, default)
        return min(int(v) if (v is not None and v == v) else default, max_val)

    def _sf(key, default):
        v = customer_row.get(key, default)
        return float(v) if (v is not None and v == v) else default

    user_cat = torch.tensor([[
        _si("agebucket_enc",           0, vocab_sizes["agebucket_enc"]           if vocab_sizes else 9999),
        _si("clubmemberstatus_enc",     0, vocab_sizes["clubmemberstatus_enc"]     if vocab_sizes else 9999),
        _si("fashionnewsfrequency_enc", 0, vocab_sizes["fashionnewsfrequency_enc"] if vocab_sizes else 9999),
    ]], dtype=torch.long).expand(n, -1)

    user_num_vals = [
        _sf("user_total_purchases",   0.0),
        _sf("user_avg_norm_price",    0.0),
        _sf("user_purchase_freq",     0.0),
        _sf("user_recency_days",    999.0),
        _sf("user_preferred_channel", 0.0),
    ]
    user_num = torch.tensor([user_num_vals], dtype=torch.float32).expand(n, -1)

    # Cross features
    # cross[:, 0] — user_ever_bought_article
    bought_set = set(user_seq["seq_article_id_enc"]) if user_seq is not None else set()
    cross = torch.zeros(n, 3, dtype=torch.float32)
    cross[:, 0] = torch.tensor(
        [1.0 if int(a) in bought_set else 0.0 for a in candidates], dtype=torch.float32
    )
    # cross[:, 1] — user_affinity_prodtype
    user_pref_pt = int(customer_row.get("user_preferred_prodtype", -1))
    cross[:, 1] = torch.tensor(
        [1.0 if pt == user_pref_pt else 0.0 for pt in pt_encs], dtype=torch.float32
    )
    # cross[:, 2] — user_price_fit
    user_avg_price = float(customer_row.get("user_avg_norm_price", 0))
    cross[:, 2] = torch.tensor(
        [abs(user_avg_price - p) for p in avg_prices], dtype=torch.float32
    )

    return {
        "seq_ids":  seq_ids_b,
        "seq_pt":   seq_pt_b,
        "seq_col":  seq_col_b,
        "seq_mask": seq_mask_b,
        "item_cat": item_cat,
        "item_num": item_num,
        "user_cat": user_cat,
        "user_num": user_num,
        "cross":    cross,
    }


def rerank_candidates(
    reranker,
    customer_row: pd.Series,
    user_seq,
    candidates: np.ndarray,
    art_features: pd.DataFrame,
    art_weekly: pd.DataFrame,
    year_week: str = None,
    batch_size: int = RERANK_BATCH,
    vocab_sizes: dict = None,
) -> np.ndarray:
    """
    Score all candidates with SASRec reranker.
    Returns candidate article_id_enc array sorted by score descending.
    """
    batch = _build_reranker_batch(
        customer_row, user_seq, candidates, art_features, art_weekly, year_week, vocab_sizes
    )

    scores = []
    n = len(candidates)
    for start in range(0, n, batch_size):
        mini = {k: v[start:start + batch_size].to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            logits = reranker(mini)
            s = torch.sigmoid(logits).cpu().numpy()
        scores.extend(s.tolist())

    scores = np.array(scores)
    order  = np.argsort(scores)[::-1]
    return candidates[order], scores[order]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: Greedy knapsack
# ══════════════════════════════════════════════════════════════════════════════

def greedy_knapsack(
    ranked_articles: np.ndarray,
    scores: np.ndarray,
    prices: dict,
    budget: float,
) -> list:
    """
    Greedy pass: add items in score order while total ≤ budget.
    Fill pass: scan remaining items to fill leftover headroom.

    Parameters
    ----------
    ranked_articles : article_id_enc array, sorted by score desc
    scores          : corresponding scores
    prices          : {article_id_enc: estimated_price_usd}
    budget          : total USD budget

    Returns
    -------
    List of article_id_enc in the basket
    """
    basket = []
    total  = 0.0

    # Greedy pass
    skipped = []
    for art, score in zip(ranked_articles, scores):
        price = prices.get(int(art), 0.0)
        if total + price <= budget:
            basket.append(int(art))
            total += price
        else:
            skipped.append((art, score, price))

    # Fill pass: try skipped items (already sorted by score)
    for art, score, price in skipped:
        if total + price <= budget:
            basket.append(int(art))
            total += price

    return basket


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def recommend(
    customer_id: str,
    budget: float,
    two_tower_model,
    reranker,
    faiss_index,
    faiss_art_ids: np.ndarray,
    user_features: pd.DataFrame,
    user_seqs: pd.DataFrame,
    art_features: pd.DataFrame,
    art_weekly: pd.DataFrame,
    vocab_sizes: dict,
    year_week: str = None,
) -> list:
    """
    Full recommendation pipeline for one customer.

    Returns list of dicts: [{article_id_enc, score, estimated_price_usd}, ...]
    sorted by score, with sum(estimated_price_usd) ≤ budget.
    """
    # Lookup user features
    cust_row_df = user_features[user_features["customer_id"] == customer_id]
    if cust_row_df.empty:
        customer_row = pd.Series(dtype=object)
    else:
        customer_row = cust_row_df.iloc[0]

    seq_df = user_seqs[user_seqs["customer_id"] == customer_id]
    user_seq = seq_df.iloc[0] if not seq_df.empty else None

    # Stage 1: retrieve
    candidates = retrieve_candidates(
        two_tower_model, faiss_index, faiss_art_ids,
        customer_row, user_seq, vocab_sizes,
    )

    # Stage 2: rerank
    ranked_arts, scores = rerank_candidates(
        reranker, customer_row, user_seq, candidates,
        art_features, art_weekly, year_week, vocab_sizes=vocab_sizes,
    )

    # Stage 3: knapsack
    prices = art_features.set_index("article_id_enc")["estimated_price_usd"].to_dict()
    basket = greedy_knapsack(ranked_arts, scores, prices, budget)

    # Build output
    art_info = art_features.set_index("article_id_enc")
    score_map = dict(zip(ranked_arts.tolist(), scores.tolist()))

    result = []
    for art in basket:
        row = art_info.loc[art] if art in art_info.index else {}
        result.append({
            "article_id_enc":      art,
            "prod_name":           row.get("prod_name", ""),
            "estimated_price_usd": float(row.get("estimated_price_usd", 0.0)),
            "score":               float(score_map.get(art, 0.0)),
        })

    result.sort(key=lambda x: x["score"], reverse=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Batch evaluation on test set
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_two_tower_only(
    two_tower_model,
    faiss_index,
    faiss_art_ids: np.ndarray,
    test_customers: list,
    test_txn: pd.DataFrame,
    uf_test: pd.DataFrame,
    user_seqs: pd.DataFrame,
    art_feat: pd.DataFrame,
    vocab_sizes: dict,
    budget: float = 100.0,
    top_k: int = 12,
):
    """Evaluate Two-Tower retrieval alone (no reranker)."""
    from model_evaluate import evaluate_pipeline, print_metrics
    import json

    recommendations = {}
    baskets         = {}
    prices          = art_feat.set_index("article_id_enc")["estimated_price_usd"].to_dict()

    # Resume from checkpoint if available
    ckpt_path = os.path.join(CHECKPOINT_DIR, "eval_tt_only.json")
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        recommendations = ckpt["recommendations"]
        baskets = {k: v for k, v in ckpt["baskets"].items()}
        start_idx = ckpt["next_idx"]
        print(f"  Resuming Two-Tower eval from customer {start_idx:,}")

    for i in range(start_idx, len(test_customers)):
        cust_id = test_customers[i]
        if i % 1000 == 0:
            print(f"  [Two-Tower only] {i:,} / {len(test_customers):,}")

        cust_row_df = uf_test[uf_test["customer_id"] == cust_id]
        customer_row = cust_row_df.iloc[0] if not cust_row_df.empty else pd.Series(dtype=object)
        seq_df = user_seqs[user_seqs["customer_id"] == cust_id]
        user_seq = seq_df.iloc[0] if not seq_df.empty else None

        candidates = retrieve_candidates(
            two_tower_model, faiss_index, faiss_art_ids,
            customer_row, user_seq, vocab_sizes, top_k=top_k,
        )

        scores = np.arange(len(candidates), 0, -1, dtype=np.float32)
        basket = greedy_knapsack(candidates, scores, prices, budget)

        recommendations[cust_id] = candidates[:top_k].tolist()
        baskets[cust_id] = basket

        # Save checkpoint every 1000 customers
        if (i + 1) % 1000 == 0:
            with open(ckpt_path, "w") as f:
                json.dump({"recommendations": recommendations, "baskets": baskets, "next_idx": i + 1}, f)

    metrics = evaluate_pipeline(
        recommendations=recommendations,
        test_txn=test_txn,
        budget=budget,
        baskets=baskets,
        art_prices=prices,
    )
    print_metrics(metrics, f"Two-Tower ONLY  (budget=${budget})")
    return metrics


def evaluate_on_test(budget: float = 100.0, skip_tt_only: bool = False):
    """Run the full pipeline on the test split and report metrics."""
    from model_evaluate import evaluate_pipeline, print_metrics

    print("Loading models and data …")
    with open(f"{INPUT_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    vocab_sizes = encoders["_vocab_sizes"]

    two_tower = load_two_tower(vocab_sizes)
    reranker  = load_reranker(vocab_sizes)
    faiss_index, faiss_art_ids = load_faiss_index()

    test_txn    = pd.read_parquet(f"{INPUT_DIR}/transactions_test.parquet")
    uf_test     = pd.read_parquet(f"{INPUT_DIR}/user_features_test.parquet")
    user_seqs   = pd.read_parquet(f"{INPUT_DIR}/user_sequences.parquet")
    art_feat    = pd.read_parquet(f"{INPUT_DIR}/article_features.parquet")
    art_weekly  = pd.read_parquet(f"{INPUT_DIR}/article_weekly_stats.parquet")
    if "article_id_enc" not in art_weekly.columns:
        id_map = art_feat[["article_id", "article_id_enc"]].drop_duplicates()
        art_weekly = art_weekly.merge(id_map, on="article_id", how="left")

    test_txn["date"] = pd.to_datetime(test_txn["date"])
    test_customers   = test_txn["customer_id"].unique().tolist()

    # Use latest training week as proxy for test-time weekly stats
    latest_week = art_weekly["year_week"].max() if "year_week" in art_weekly.columns else None

    # Subsample test customers for faster evaluation
    MAX_TEST_CUSTOMERS = 50000
    if len(test_customers) > MAX_TEST_CUSTOMERS:
        rng = np.random.default_rng(42)
        test_customers = rng.choice(test_customers, MAX_TEST_CUSTOMERS, replace=False).tolist()

    # Filter test_txn to only subsampled customers so ground truth matches
    test_cust_set = set(test_customers)
    test_txn = test_txn[test_txn["customer_id"].isin(test_cust_set)].copy()

    print(f"Evaluating on {len(test_customers):,} test customers …")

    # Pre-index for O(1) lookup — cold-start users simply won't appear in these dicts
    uf_index  = uf_test.set_index("customer_id")
    seq_index = user_seqs.set_index("customer_id") if "customer_id" in user_seqs.columns else user_seqs

    # ── Evaluation 1: Two-Tower only ─────────────────────────────────────────
    if not skip_tt_only:
        print("\n--- Two-Tower Only ---")
        tt_metrics = evaluate_two_tower_only(
            two_tower, faiss_index, faiss_art_ids,
            test_customers, test_txn, uf_test, user_seqs, art_feat,
            vocab_sizes, budget=budget,
        )

    # ── Evaluation 2: Two-Tower + Reranker ───────────────────────────────────
    print("\n--- Two-Tower + Reranker ---")
    import json

    recommendations = {}
    baskets         = {}
    prices          = art_feat.set_index("article_id_enc")["estimated_price_usd"].to_dict()

    # Resume from checkpoint if available
    ckpt_path = os.path.join(CHECKPOINT_DIR, "eval_full.json")
    start_idx = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        recommendations = ckpt["recommendations"]
        baskets = {k: v for k, v in ckpt["baskets"].items()}
        start_idx = ckpt["next_idx"]
        print(f"  Resuming full pipeline eval from customer {start_idx:,}")

    for i in range(start_idx, len(test_customers)):
        cust_id = test_customers[i]
        if i % 1000 == 0:
            print(f"  [Full pipeline] {i:,} / {len(test_customers):,}")

        # Get user features — returns empty Series / None for cold-start users
        customer_row = uf_index.loc[cust_id] if cust_id in uf_index.index else pd.Series(dtype=object)
        user_seq     = seq_index.loc[cust_id] if cust_id in seq_index.index else None

        # Stage 1: retrieve top-500
        candidates = retrieve_candidates(
            two_tower, faiss_index, faiss_art_ids,
            customer_row, user_seq, vocab_sizes,
        )

        # Stage 2: rerank all 500 candidates
        ranked_arts, scores = rerank_candidates(
            reranker, customer_row, user_seq, candidates,
            art_feat, art_weekly, year_week=latest_week, vocab_sizes=vocab_sizes,
        )

        # For metrics: use top-12 from reranked list (fair comparison with two-tower)
        recommendations[cust_id] = ranked_arts[:12].tolist()

        # For budget util: apply knapsack to full reranked list
        basket = greedy_knapsack(ranked_arts, scores, prices, budget)
        baskets[cust_id] = basket

        # Save checkpoint every 1000 customers
        if (i + 1) % 1000 == 0:
            with open(ckpt_path, "w") as f:
                json.dump({"recommendations": recommendations, "baskets": baskets, "next_idx": i + 1}, f)

    full_metrics = evaluate_pipeline(
        recommendations=recommendations,
        test_txn=test_txn,
        budget=budget,
        baskets=baskets,
        art_prices=prices,
    )
    print_metrics(full_metrics, f"Two-Tower + SASRec  (budget=${budget})")

    # ── Comparison ───────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"  COMPARISON: Reranker improvement")
    print(f"{'═'*50}")
    for key in ["map@12", "ndcg@12", "hr@12"]:
        tt_val   = tt_metrics.get(key, 0)
        full_val = full_metrics.get(key, 0)
        diff     = full_val - tt_val
        pct      = (diff / tt_val * 100) if tt_val > 0 else 0
        print(f"  {key:15s}  TT={tt_val:.4f}  Full={full_val:.4f}  Δ={diff:+.4f} ({pct:+.1f}%)")
    print(f"{'═'*50}")

    return full_metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget",      type=float, default=100.0, help="Budget in USD")
    parser.add_argument("--customer_id", type=str,   default=None, help="Single customer to score")
    parser.add_argument("--evaluate",    action="store_true",      help="Run full test-set evaluation")
    parser.add_argument("--skip_tt_only", action="store_true",     help="Skip two-tower-only evaluation")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_on_test(budget=args.budget, skip_tt_only=args.skip_tt_only)
        return

    if args.customer_id:
        with open(f"{INPUT_DIR}/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        vocab_sizes = encoders["_vocab_sizes"]

        two_tower   = load_two_tower(vocab_sizes)
        reranker    = load_reranker(vocab_sizes)
        faiss_index, faiss_art_ids = load_faiss_index()

        uf_test   = pd.read_parquet(f"{INPUT_DIR}/user_features_test.parquet")
        user_seqs = pd.read_parquet(f"{INPUT_DIR}/user_sequences.parquet")
        art_feat  = pd.read_parquet(f"{INPUT_DIR}/article_features.parquet")
        art_weekly = pd.read_parquet(f"{INPUT_DIR}/article_weekly_stats.parquet")
        if "article_id_enc" not in art_weekly.columns:
            id_map = art_feat[["article_id", "article_id_enc"]].drop_duplicates()
            art_weekly = art_weekly.merge(id_map, on="article_id", how="left")

        result = recommend(
            customer_id=args.customer_id,
            budget=args.budget,
            two_tower_model=two_tower,
            reranker=reranker,
            faiss_index=faiss_index,
            faiss_art_ids=faiss_art_ids,
            user_features=uf_test,
            user_seqs=user_seqs,
            art_features=art_feat,
            art_weekly=art_weekly,
            vocab_sizes=vocab_sizes,
        )
        total = sum(r["estimated_price_usd"] for r in result)
        print(f"\nRecommendations for {args.customer_id}  (budget=${args.budget:.2f}):")
        print(f"{'Rank':<5} {'Article':<12} {'Score':<8} {'Price':>8}  Name")
        print("─" * 70)
        for i, r in enumerate(result, 1):
            print(f"{i:<5} {r['article_id_enc']:<12} {r['score']:<8.4f} "
                  f"${r['estimated_price_usd']:>7.2f}  {r['prod_name'][:40]}")
        print(f"\nBasket total: ${total:.2f} / ${args.budget:.2f}")
    else:
        print("Usage: python inference.py --customer_id <id> --budget 100")
        print("       python inference.py --evaluate --budget 100")


if __name__ == "__main__":
    main()
