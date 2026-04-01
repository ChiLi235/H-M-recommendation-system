"""
two_tower_model.py

Two-Tower retrieval model for candidate generation.

User Tower:
  Embeddings: age_bucket, club_member_status, fashion_news_frequency
  Mean-pool over last-20 purchase article embeddings
  2 dense layers (256 → 128) → L2-normalised 128-dim user embedding

Item Tower:
  Embeddings: product_type, index_group, colour_group, garment_group
  Numerics (batch-normed): normalised_price, log_global_sales, channel1_ratio
  2 dense layers (256 → 128) → L2-normalised 128-dim item embedding

Training: in-batch sampled softmax (dot-product similarity matrix, diagonal = positives)
Inference: pre-compute all item embeddings → FAISS IVF flat index → ANN top-500 per user
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

INPUT_DIR      = os.environ.get("SM_CHANNEL_PROCESSED", "processed")
OUTPUT_DIR     = os.environ.get("SM_OUTPUT_DATA_DIR", "processed")
MODEL_DIR      = os.environ.get("SM_MODEL_DIR", "models")
CHECKPOINT_DIR = "/opt/ml/checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMB_DIM     = 32
HIDDEN      = 256
OUTPUT_DIM  = 128
MAX_SEQ_LEN = 20
BATCH_SIZE  = 1024
EPOCHS      = 15
LR          = 1e-3
TEMPERATURE = 0.07
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TwoTowerDataset(Dataset):
    """
    Loads two_tower_{split}.parquet.
    Each row = one positive (user, item) pair.
    In-batch negatives are handled by the training loop.
    """

    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)
        self._validate()

    def _validate(self):
        required = [
            "customer_id_enc", "agebucket_enc", "clubmemberstatus_enc",
            "fashionnewsfrequency_enc", "user_total_purchases", "user_avg_norm_price",
            "user_purchase_freq", "user_recency_days", "user_preferred_channel",
            "article_id_enc", "producttype_enc", "indexgroup_enc",
            "colourgroup_enc", "garmentgroup_enc",
            "article_avg_norm_price", "log_global_sales", "article_channel1_ratio",
            "seq_article_id_enc", "seq_len",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            print(f"  WARNING: missing columns {missing} — filling with 0")
            for c in missing:
                self.df[c] = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── user inputs ───────────────────────────────────────────────────────
        age_enc   = int(row["agebucket_enc"])
        club_enc  = int(row["clubmemberstatus_enc"])
        news_enc  = int(row["fashionnewsfrequency_enc"])

        user_num = torch.nan_to_num(torch.tensor([
            float(row["user_total_purchases"]),
            float(row["user_avg_norm_price"]),
            float(row["user_purchase_freq"]),
            float(row["user_recency_days"]),
            float(row["user_preferred_channel"]),
        ], dtype=torch.float32))

        # sequence: pad / truncate to MAX_SEQ_LEN
        seq = row["seq_article_id_enc"]
        seq = seq[-MAX_SEQ_LEN:] if len(seq) > MAX_SEQ_LEN else seq
        seq_len = len(seq)
        seq_ids = torch.zeros(MAX_SEQ_LEN, dtype=torch.long)
        if seq_len > 0:
            seq_ids[:seq_len] = torch.tensor(seq, dtype=torch.long)
        seq_mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        seq_mask[:seq_len] = True

        # ── item inputs ───────────────────────────────────────────────────────
        item_id  = int(row["article_id_enc"])
        pt_enc   = int(row["producttype_enc"])
        ig_enc   = int(row["indexgroup_enc"])
        cg_enc   = int(row["colourgroup_enc"])
        gg_enc   = int(row["garmentgroup_enc"])

        item_num = torch.nan_to_num(torch.tensor([
            float(row["article_avg_norm_price"]),
            float(row["log_global_sales"]),
            float(row["article_channel1_ratio"]),
        ], dtype=torch.float32))

        return {
            "age_enc":   torch.tensor(age_enc,  dtype=torch.long),
            "club_enc":  torch.tensor(club_enc, dtype=torch.long),
            "news_enc":  torch.tensor(news_enc, dtype=torch.long),
            "user_num":  user_num,
            "seq_ids":   seq_ids,
            "seq_mask":  seq_mask,
            "item_id":   torch.tensor(item_id, dtype=torch.long),
            "pt_enc":    torch.tensor(pt_enc,  dtype=torch.long),
            "ig_enc":    torch.tensor(ig_enc,  dtype=torch.long),
            "cg_enc":    torch.tensor(cg_enc,  dtype=torch.long),
            "gg_enc":    torch.tensor(gg_enc,  dtype=torch.long),
            "item_num":  item_num,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class UserTower(nn.Module):
    def __init__(self, vocab_sizes: dict, emb_dim=EMB_DIM, hidden=HIDDEN, out_dim=OUTPUT_DIM):
        super().__init__()
        self.age_emb  = nn.Embedding(vocab_sizes["agebucket_enc"] + 1,           emb_dim)
        self.club_emb = nn.Embedding(vocab_sizes["clubmemberstatus_enc"] + 1,    emb_dim)
        self.news_emb = nn.Embedding(vocab_sizes["fashionnewsfrequency_enc"] + 1, emb_dim)
        # Shared article embedding used for sequence mean-pooling
        self.art_emb  = nn.Embedding(vocab_sizes["article_id_enc"] + 1, emb_dim, padding_idx=0)

        in_dim = 3 * emb_dim + emb_dim + 5   # 3 cat embs + seq pool + 5 numerics
        self.input_norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, age_enc, club_enc, news_enc, user_num, seq_ids, seq_mask):
        age  = self.age_emb(age_enc)   # (B, emb)
        club = self.club_emb(club_enc)
        news = self.news_emb(news_enc)

        # sequence mean pooling (ignore padding)
        embs = self.art_emb(seq_ids)          # (B, L, emb)
        mask_f = seq_mask.float().unsqueeze(-1)  # (B, L, 1)
        denom  = mask_f.sum(1).clamp(min=1)
        seq_pool = (embs * mask_f).sum(1) / denom  # (B, emb)

        x = torch.cat([age, club, news, seq_pool, user_num], dim=-1)
        x = self.input_norm(x)
        return F.normalize(self.net(x), dim=-1, eps=1e-8)


class ItemTower(nn.Module):
    def __init__(self, vocab_sizes: dict, emb_dim=EMB_DIM, hidden=HIDDEN, out_dim=OUTPUT_DIM):
        super().__init__()
        self.pt_emb = nn.Embedding(vocab_sizes["producttype_enc"] + 1,  emb_dim)
        self.ig_emb = nn.Embedding(vocab_sizes["indexgroup_enc"] + 1,   emb_dim)
        self.cg_emb = nn.Embedding(vocab_sizes["colourgroup_enc"] + 1,  emb_dim)
        self.gg_emb = nn.Embedding(vocab_sizes["garmentgroup_enc"] + 1, emb_dim)

        in_dim = 4 * emb_dim + 3   # 4 cat embs + 3 numerics
        self.input_norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pt_enc, ig_enc, cg_enc, gg_enc, item_num):
        pt = self.pt_emb(pt_enc)
        ig = self.ig_emb(ig_enc)
        cg = self.cg_emb(cg_enc)
        gg = self.gg_emb(gg_enc)
        x = torch.cat([pt, ig, cg, gg, item_num], dim=-1)
        x = self.input_norm(x)
        return F.normalize(self.net(x), dim=-1, eps=1e-8)


class TwoTowerModel(nn.Module):
    def __init__(self, vocab_sizes: dict):
        super().__init__()
        self.user_tower = UserTower(vocab_sizes)
        self.item_tower = ItemTower(vocab_sizes)

    def user_embed(self, batch):
        return self.user_tower(
            batch["age_enc"], batch["club_enc"], batch["news_enc"],
            batch["user_num"], batch["seq_ids"], batch["seq_mask"],
        )

    def item_embed(self, batch):
        return self.item_tower(
            batch["pt_enc"], batch["ig_enc"], batch["cg_enc"],
            batch["gg_enc"], batch["item_num"],
        )

    def forward(self, batch):
        user_emb = self.user_embed(batch)
        item_emb = self.item_embed(batch)
        logits = user_emb @ item_emb.T  # (B, B)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# In-batch softmax loss
# ══════════════════════════════════════════════════════════════════════════════

def in_batch_softmax_loss(logits: torch.Tensor, temperature: float = TEMPERATURE):
    """
    logits: (B, B) dot-product matrix; diagonal is positive (user i bought item i).
    Returns scalar cross-entropy loss.
    """
    B = logits.shape[0]
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits / temperature, labels)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    start_epoch   = 1

    # ── resume from spot checkpoint if available ───────────────────────────────
    resume_path = os.path.join(CHECKPOINT_DIR, "two_tower_resume.pt")
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"  Resumed at epoch {start_epoch}  best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch)
            loss   = in_batch_softmax_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(batch)
                val_loss += in_batch_softmax_loss(logits).item()
                val_batches += 1
        val_loss /= max(val_batches, 1)

        scheduler.step()
        print(f"  Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ── save best model weights ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower_best.pt"))
            print(f"    ✓ best model saved  (val_loss={val_loss:.4f})")

        # ── save spot resume checkpoint every epoch ────────────────────────────
        torch.save({
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, resume_path)

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# FAISS index builder
# ══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(
    model: TwoTowerModel,
    art_features: pd.DataFrame,
    batch_size: int = 4096,
):
    """
    Computes 128-dim embeddings for all articles and builds a FAISS flat index.
    Returns (index, article_id_enc_array) so query results can be mapped back to article IDs.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    print("Building FAISS index …")
    model.eval()
    model.to(DEVICE)

    art_sel = art_features[
        ["article_id_enc", "producttype_enc", "indexgroup_enc",
         "colourgroup_enc", "garmentgroup_enc",
         "article_avg_norm_price", "log_global_sales", "article_channel1_ratio"]
    ].drop_duplicates("article_id_enc").reset_index(drop=True)

    all_embs = []
    for start in range(0, len(art_sel), batch_size):
        chunk = art_sel.iloc[start:start + batch_size]
        batch = {
            "pt_enc":   torch.tensor(chunk["producttype_enc"].values,  dtype=torch.long).to(DEVICE),
            "ig_enc":   torch.tensor(chunk["indexgroup_enc"].values,    dtype=torch.long).to(DEVICE),
            "cg_enc":   torch.tensor(chunk["colourgroup_enc"].values,   dtype=torch.long).to(DEVICE),
            "gg_enc":   torch.tensor(chunk["garmentgroup_enc"].values,  dtype=torch.long).to(DEVICE),
            "item_num": torch.tensor(
                chunk[["article_avg_norm_price", "log_global_sales", "article_channel1_ratio"]].values,
                dtype=torch.float32,
            ).to(DEVICE),
        }
        with torch.no_grad():
            emb = model.item_embed(batch).cpu().numpy()
        all_embs.append(emb)

    embs = np.vstack(all_embs).astype(np.float32)
    article_ids = art_sel["article_id_enc"].values

    # Inner-product index (embeddings are L2-normalised → equivalent to cosine similarity)
    index = faiss.IndexFlatIP(OUTPUT_DIM)
    index.add(embs)
    print(f"  FAISS index: {index.ntotal:,} vectors, dim={OUTPUT_DIM}")

    return index, article_ids


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    args, _ = parser.parse_known_args()

    # ── load vocab sizes ──────────────────────────────────────────────────────
    with open(f"{INPUT_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    vocab_sizes = encoders["_vocab_sizes"]

    # ── datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = TwoTowerDataset(f"{INPUT_DIR}/two_tower_train.parquet")
    val_ds   = TwoTowerDataset(f"{INPUT_DIR}/two_tower_val.parquet")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Train: {len(train_ds):,} rows  Val: {len(val_ds):,} rows")

    # ── model ─────────────────────────────────────────────────────────────────
    model = TwoTowerModel(vocab_sizes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters  Device: {DEVICE}")
    print(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # ── train ─────────────────────────────────────────────────────────────────
    model = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    # load best checkpoint
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "two_tower_best.pt")))

    # ── save final model ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "two_tower_final.pt"))
    print(f"Saved two_tower_final.pt")

    # ── build FAISS index ─────────────────────────────────────────────────────
    art_feat = pd.read_parquet(f"{INPUT_DIR}/article_features.parquet")
    index, article_ids = build_faiss_index(model, art_feat)

    import faiss
    faiss.write_index(index, os.path.join(MODEL_DIR, "faiss_index.bin"))
    np.save(os.path.join(MODEL_DIR, "faiss_article_ids.npy"), article_ids)
    print(f"Saved faiss_index.bin and faiss_article_ids.npy")


if __name__ == "__main__":
    main()
