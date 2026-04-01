"""
train_pipeline.py

End-to-end training orchestration:
  Step 1  data_preprocessing.py       (if processed/ is missing)
  Step 2  feature_engineering.py      (if article/user features are missing)
  Step 3  training_dataset_builder.py (if two_tower / reranker parquets are missing)
  Step 4  two_tower_model.py          train Two-Tower, build FAISS index
  Step 5  reranker_model.py           train SASRec reranker
  Step 6  model_evaluate.py           run baselines + pipeline evaluation

Each step is skipped if its outputs already exist (re-runnable / resumable).

Run:
  python train_pipeline.py
  python train_pipeline.py --force_retrain     # re-run all steps
  python train_pipeline.py --skip_preprocess   # skip steps 1-3 (data already built)
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

INPUT_DIR  = "processed"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── hyperparameters (override defaults in each module) ─────────────────────────
BATCH_SIZE_TT     = 1024
BATCH_SIZE_RER    = 256
EPOCHS_TT         = 15
EPOCHS_RER        = 10
LR                = 1e-3
NUM_WORKERS       = 4


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _all_exist(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s//60}m{s%60:02d}s"


def _step_header(n: int, name: str):
    print(f"\n{'═'*60}")
    print(f"  STEP {n}: {name}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 & 2: preprocessing + feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing(force: bool = False):
    outputs = [
        f"{INPUT_DIR}/transactions_train.parquet",
        f"{INPUT_DIR}/articles_encoded.parquet",
        f"{INPUT_DIR}/encoders.pkl",
    ]
    if not force and _all_exist(*outputs):
        print("  Preprocessing outputs already exist — skipping.")
        return

    _step_header(1, "Data Preprocessing")
    t0 = time.time()
    import data_preprocessing
    data_preprocessing.main()
    print(f"  Done in {_elapsed(t0)}")


def run_feature_engineering(force: bool = False):
    outputs = [
        f"{INPUT_DIR}/article_features.parquet",
        f"{INPUT_DIR}/user_sequences.parquet",
        f"{INPUT_DIR}/user_features_train.parquet",
    ]
    if not force and _all_exist(*outputs):
        print("  Feature engineering outputs already exist — skipping.")
        return

    _step_header(2, "Feature Engineering")
    t0 = time.time()
    import feature_engineering
    feature_engineering.main()
    print(f"  Done in {_elapsed(t0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: training dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset_builder(force: bool = False):
    outputs = [
        f"{INPUT_DIR}/two_tower_train.parquet",
        f"{INPUT_DIR}/reranker_train.parquet",
    ]
    if not force and _all_exist(*outputs):
        print("  Training dataset files already exist — skipping.")
        return

    _step_header(3, "Training Dataset Builder")
    t0 = time.time()
    import training_dataset_builder
    training_dataset_builder.main()
    print(f"  Done in {_elapsed(t0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Two-Tower training + FAISS index
# ══════════════════════════════════════════════════════════════════════════════

def run_two_tower(force: bool = False):
    outputs = [
        os.path.join(MODEL_DIR, "two_tower_final.pt"),
        os.path.join(MODEL_DIR, "faiss_index.bin"),
    ]
    if not force and _all_exist(*outputs):
        print("  Two-Tower model already exists — skipping.")
        return

    _step_header(4, "Two-Tower Training")
    t0 = time.time()

    from two_tower_model import (
        TwoTowerDataset, TwoTowerModel,
        train as tt_train, build_faiss_index,
        EPOCHS, LR, BATCH_SIZE, NUM_WORKERS, OUTPUT_DIM,
    )
    import faiss

    with open(f"{INPUT_DIR}/encoders.pkl", "rb") as f:
        vocab_sizes = pickle.load(f)["_vocab_sizes"]

    train_ds = TwoTowerDataset(f"{INPUT_DIR}/two_tower_train.parquet")
    val_ds   = TwoTowerDataset(f"{INPUT_DIR}/two_tower_val.parquet")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TwoTowerModel(vocab_sizes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Two-Tower: {n_params:,} params  device={device}")

    model = tt_train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower_best.pt"),
                                     map_location=device))
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower_final.pt"))

    # FAISS index
    art_feat = pd.read_parquet(f"{INPUT_DIR}/article_features.parquet")
    index, art_ids = build_faiss_index(model, art_feat)
    faiss.write_index(index, os.path.join(MODEL_DIR, "faiss_index.bin"))
    np.save(os.path.join(MODEL_DIR, "faiss_article_ids.npy"), art_ids)

    print(f"  Done in {_elapsed(t0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: SASRec reranker training
# ══════════════════════════════════════════════════════════════════════════════

def run_reranker(force: bool = False):
    output = os.path.join(MODEL_DIR, "reranker_final.pt")
    if not force and os.path.exists(output):
        print("  Reranker model already exists — skipping.")
        return

    _step_header(5, "SASRec Reranker Training")
    t0 = time.time()

    from reranker_model import (
        RerankerDataset, SASRecReranker,
        train as rer_train,
        EPOCHS, LR, BATCH_SIZE, NUM_WORKERS,
    )

    with open(f"{INPUT_DIR}/encoders.pkl", "rb") as f:
        vocab_sizes = pickle.load(f)["_vocab_sizes"]

    train_ds = RerankerDataset(f"{INPUT_DIR}/reranker_train.parquet")
    val_ds   = RerankerDataset(f"{INPUT_DIR}/reranker_val.parquet")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SASRecReranker(vocab_sizes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  SASRec: {n_params:,} params  device={device}")

    model = rer_train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "reranker_best.pt"),
                                     map_location=device))
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "reranker_final.pt"))

    print(f"  Done in {_elapsed(t0)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(budget: float = 50.0):
    _step_header(6, "Evaluation")

    # Baselines first (cheap)
    from model_evaluate import main as eval_baselines
    eval_baselines()

    # Full pipeline evaluation
    from inference import evaluate_on_test
    metrics = evaluate_on_test(budget=budget)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train the H&M recommendation pipeline")
    parser.add_argument("--force_retrain",   action="store_true", help="Re-run all steps")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing steps 1-3")
    parser.add_argument("--budget",          type=float, default=50.0, help="Budget (USD) for evaluation")
    parser.add_argument("--eval_only",       action="store_true", help="Skip training, run evaluation only")
    args = parser.parse_args()

    total_t0 = time.time()
    print(f"\n{'═'*60}")
    print(f"  H&M Budget-Constrained Recommendation Pipeline")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'═'*60}")

    if not args.eval_only:
        if not args.skip_preprocess:
            run_preprocessing(force=args.force_retrain)
            run_feature_engineering(force=args.force_retrain)
            run_dataset_builder(force=args.force_retrain)

        run_two_tower(force=args.force_retrain)
        run_reranker(force=args.force_retrain)

    run_evaluation(budget=args.budget)

    print(f"\n{'═'*60}")
    print(f"  Pipeline complete in {_elapsed(total_t0)}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
