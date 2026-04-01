"""
reranker_model.py

SASRec-style reranker: scores top-500 Two-Tower candidates per user.

Architecture (cost-optimised for ml.g4dn.xlarge):
  - Sequence encoder: 1 self-attention layer, 2 heads, 96-dim (64 art_id + 16 prodtype + 16 colour)
  - Candidate features: product_type, index_group, colour_group, garment_group embeddings + numerics
  - User demographics: age_bucket, club_member, fashion_news embeddings + behavioural numerics
  - Cross features: ever_bought, affinity_prodtype, price_fit, sales_last_4weeks, sales_last_8weeks
  - MLP: (seq_dim + cand_dim + user_dim + cross_dim) → 256 → 64 → 1

Training: binary cross-entropy, positives:negatives ≈ 1:5 (hard popularity-based negatives)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

INPUT_DIR      = os.environ.get("SM_CHANNEL_PROCESSED", "processed")
OUTPUT_DIR     = os.environ.get("SM_OUTPUT_DATA_DIR", "processed")
MODEL_DIR      = os.environ.get("SM_MODEL_DIR", "models")
CHECKPOINT_DIR = "/opt/ml/checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_EMB_DIM  = 64    # article_id embedding dim in sequence
CAT_EMB_DIM  = 16    # dim for product_type / colour in sequence
ITEM_EMB_DIM = 24    # dim for candidate item categoricals
USER_EMB_DIM = 16    # dim for user demographic categoricals
MAX_SEQ_LEN  = 20
N_HEADS      = 2
HIDDEN       = 256
DROPOUT      = 0.1
BATCH_SIZE   = 256
EPOCHS       = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class RerankerDataset(Dataset):
    """
    Loads reranker_{split}.parquet.
    Each row = one (user_sequence, candidate_article, label) sample.
    """

    SEQ_COLS = ["seq_article_id_enc", "seq_producttype_enc", "seq_colourgroup_enc"]

    ITEM_NUM_COLS = [
        "article_avg_norm_price", "log_global_sales", "article_channel1_ratio",
        "sales_last_4weeks", "sales_last_8weeks",
    ]
    USER_NUM_COLS = [
        "user_total_purchases", "user_avg_norm_price", "user_purchase_freq",
        "user_recency_days", "user_preferred_channel",
    ]
    CROSS_COLS = [
        "user_ever_bought_article", "user_affinity_prodtype", "user_price_fit",
    ]

    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)
        # fill any missing numeric columns
        for col in self.ITEM_NUM_COLS + self.USER_NUM_COLS + self.CROSS_COLS:
            if col not in self.df.columns:
                self.df[col] = 0.0
            else:
                self.df[col] = self.df[col].fillna(0.0)
        for col in ["seq_len"]:
            if col not in self.df.columns:
                self.df[col] = 0
        for col in self.SEQ_COLS:
            if col not in self.df.columns:
                self.df[col] = [[] for _ in range(len(self.df))]

        print(f"  RerankerDataset: {len(self.df):,} rows  "
              f"pos={self.df['label'].sum():,}  "
              f"neg={(self.df['label']==0).sum():,}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        # ── sequence ──────────────────────────────────────────────────────────
        def _pad_seq(lst, maxlen):
            lst = list(lst)[-maxlen:] if len(lst) > maxlen else list(lst)
            t = torch.zeros(maxlen, dtype=torch.long)
            if lst:
                t[:len(lst)] = torch.tensor(lst, dtype=torch.long)
            return t

        seq_ids  = _pad_seq(row["seq_article_id_enc"],  MAX_SEQ_LEN)
        seq_pt   = _pad_seq(row["seq_producttype_enc"], MAX_SEQ_LEN)
        seq_col  = _pad_seq(row["seq_colourgroup_enc"], MAX_SEQ_LEN)

        seq_len  = min(int(row["seq_len"]), MAX_SEQ_LEN)
        seq_mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        seq_mask[:seq_len] = True

        # ── candidate item ────────────────────────────────────────────────────
        item_cat = torch.tensor([
            int(row.get("producttype_enc",  0)),
            int(row.get("indexgroup_enc",   0)),
            int(row.get("colourgroup_enc",  0)),
            int(row.get("garmentgroup_enc", 0)),
        ], dtype=torch.long)

        item_num = torch.nan_to_num(torch.tensor(
            [float(row[c]) for c in self.ITEM_NUM_COLS], dtype=torch.float32
        ), nan=0.0)

        # ── user demographics ─────────────────────────────────────────────────
        user_cat = torch.tensor([
            int(row.get("agebucket_enc",           0)),
            int(row.get("clubmemberstatus_enc",     0)),
            int(row.get("fashionnewsfrequency_enc", 0)),
        ], dtype=torch.long)

        user_num = torch.nan_to_num(torch.tensor(
            [float(row[c]) for c in self.USER_NUM_COLS], dtype=torch.float32
        ), nan=0.0)

        # ── cross features ────────────────────────────────────────────────────
        cross = torch.nan_to_num(torch.tensor(
            [float(row[c]) for c in self.CROSS_COLS], dtype=torch.float32
        ), nan=0.0)

        return {
            "seq_ids":  seq_ids,
            "seq_pt":   seq_pt,
            "seq_col":  seq_col,
            "seq_mask": seq_mask,
            "item_cat": item_cat,
            "item_num": item_num,
            "user_cat": user_cat,
            "user_num": user_num,
            "cross":    cross,
            "label":    label,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class SASRecReranker(nn.Module):
    """
    Self-attentive sequential reranker.

    seq_dim  = SEQ_EMB_DIM + CAT_EMB_DIM + CAT_EMB_DIM = 64 + 16 + 16 = 96
    cand_dim = 4 * ITEM_EMB_DIM + 5 (numerics)
    user_dim = 3 * USER_EMB_DIM + 5 (numerics)
    cross_dim = 3
    """

    def __init__(self, vocab_sizes: dict):
        super().__init__()

        seq_dim  = SEQ_EMB_DIM + 2 * CAT_EMB_DIM
        cand_dim = 4 * ITEM_EMB_DIM + len(RerankerDataset.ITEM_NUM_COLS)
        user_dim = 3 * USER_EMB_DIM + len(RerankerDataset.USER_NUM_COLS)
        cross_dim = len(RerankerDataset.CROSS_COLS)

        # ── sequence embeddings ───────────────────────────────────────────────
        self.art_emb  = nn.Embedding(vocab_sizes["article_id_enc"] + 1,
                                     SEQ_EMB_DIM, padding_idx=0)
        self.spt_emb  = nn.Embedding(vocab_sizes["producttype_enc"] + 1,
                                     CAT_EMB_DIM, padding_idx=0)
        self.scol_emb = nn.Embedding(vocab_sizes["colourgroup_enc"] + 1,
                                     CAT_EMB_DIM, padding_idx=0)
        self.pos_emb  = nn.Embedding(MAX_SEQ_LEN + 1, seq_dim)

        # ── self-attention (1 layer, 2 heads) ─────────────────────────────────
        self.attn_norm = nn.LayerNorm(seq_dim)
        self.attn      = nn.MultiheadAttention(seq_dim, N_HEADS, dropout=DROPOUT,
                                               batch_first=True)
        self.ffn_norm  = nn.LayerNorm(seq_dim)
        self.ffn = nn.Sequential(
            nn.Linear(seq_dim, seq_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(seq_dim * 2, seq_dim),
        )

        # ── candidate item embeddings ─────────────────────────────────────────
        self.ipt_emb = nn.Embedding(vocab_sizes["producttype_enc"] + 1,  ITEM_EMB_DIM)
        self.iig_emb = nn.Embedding(vocab_sizes["indexgroup_enc"] + 1,   ITEM_EMB_DIM)
        self.icg_emb = nn.Embedding(vocab_sizes["colourgroup_enc"] + 1,  ITEM_EMB_DIM)
        self.igg_emb = nn.Embedding(vocab_sizes["garmentgroup_enc"] + 1, ITEM_EMB_DIM)
        self.item_num_norm = nn.LayerNorm(len(RerankerDataset.ITEM_NUM_COLS))

        # ── user demographic embeddings ───────────────────────────────────────
        self.uage_emb  = nn.Embedding(vocab_sizes["agebucket_enc"] + 1,           USER_EMB_DIM)
        self.uclub_emb = nn.Embedding(vocab_sizes["clubmemberstatus_enc"] + 1,    USER_EMB_DIM)
        self.unews_emb = nn.Embedding(vocab_sizes["fashionnewsfrequency_enc"] + 1, USER_EMB_DIM)
        self.user_num_norm = nn.LayerNorm(len(RerankerDataset.USER_NUM_COLS))

        # ── MLP scorer ────────────────────────────────────────────────────────
        mlp_in = seq_dim + cand_dim + user_dim + cross_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _encode_sequence(self, seq_ids, seq_pt, seq_col, seq_mask):
        """Self-attention over purchase sequence → last valid hidden state."""
        # Token embeddings
        x = torch.cat([
            self.art_emb(seq_ids),  # (B, L, SEQ_EMB_DIM)
            self.spt_emb(seq_pt),   # (B, L, CAT_EMB_DIM)
            self.scol_emb(seq_col), # (B, L, CAT_EMB_DIM)
        ], dim=-1)                  # (B, L, seq_dim)

        # Positional embeddings
        B, L, _ = x.shape
        pos = torch.arange(1, L + 1, device=x.device).unsqueeze(0).expand(B, -1)
        pos = pos * seq_mask.long()  # zero-out padding positions
        x = x + self.pos_emb(pos)

        # Causal self-attention mask (upper-triangular → future positions masked)
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Layer norm → attention → residual
        x_norm = self.attn_norm(x)
        # Ensure at least position 0 is always unmasked — prevents all-True
        # key_padding_mask for users with seq_len=0 (softmax over all -inf = NaN)
        safe_mask = seq_mask.clone()
        safe_mask[:, 0] = True
        key_padding_mask = ~safe_mask  # True = ignore (padding)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                attn_mask=causal_mask,
                                key_padding_mask=key_padding_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + attn_out

        # FFN → residual
        x = x + self.ffn(self.ffn_norm(x))

        # Extract the last non-padding position for each sample
        seq_lengths = seq_mask.long().sum(dim=1).clamp(min=1) - 1  # 0-indexed last valid
        seq_repr = x[torch.arange(B, device=x.device), seq_lengths]  # (B, seq_dim)
        return torch.nan_to_num(seq_repr, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, batch):
        # Sequence representation
        seq_repr = self._encode_sequence(
            batch["seq_ids"], batch["seq_pt"], batch["seq_col"], batch["seq_mask"]
        )

        # Candidate item features
        ipt = self.ipt_emb(batch["item_cat"][:, 0])
        iig = self.iig_emb(batch["item_cat"][:, 1])
        icg = self.icg_emb(batch["item_cat"][:, 2])
        igg = self.igg_emb(batch["item_cat"][:, 3])
        inum = self.item_num_norm(batch["item_num"])
        cand_feat = torch.cat([ipt, iig, icg, igg, inum], dim=-1)

        # User demographic features
        uage  = self.uage_emb(batch["user_cat"][:, 0])
        uclub = self.uclub_emb(batch["user_cat"][:, 1])
        unews = self.unews_emb(batch["user_cat"][:, 2])
        unum  = self.user_num_norm(batch["user_num"])
        user_feat = torch.cat([uage, uclub, unews, unum], dim=-1)

        # Concatenate all features and score
        x = torch.cat([seq_repr, cand_feat, user_feat, batch["cross"]], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, -10.0, 10.0)
        return self.mlp(x).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model: SASRecReranker,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
):
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    start_epoch   = 1

    # ── resume from spot checkpoint if available ───────────────────────────────
    resume_path = os.path.join(CHECKPOINT_DIR, "reranker_resume.pt")
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
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            if torch.isnan(loss) or torch.isinf(loss):
                continue  # skip bad batch
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(batch)
                bloss = F.binary_cross_entropy_with_logits(logits, batch["label"])
                if not (torch.isnan(bloss) or torch.isinf(bloss)):
                    val_loss += bloss.item()
                    val_batches += 1
        val_loss /= max(val_batches, 1)

        scheduler.step()
        print(f"  Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # ── save best model weights ────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "reranker_best.pt"))
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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    args, _ = parser.parse_known_args()

    with open(f"{INPUT_DIR}/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    vocab_sizes = encoders["_vocab_sizes"]

    print("Loading datasets …")
    train_ds = RerankerDataset(f"{INPUT_DIR}/reranker_train.parquet")
    val_ds   = RerankerDataset(f"{INPUT_DIR}/reranker_val.parquet")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = SASRecReranker(vocab_sizes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters  Device: {DEVICE}")
    print(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    model = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    best_path = os.path.join(MODEL_DIR, "reranker_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        print("Loaded best checkpoint")
    else:
        print("Warning: reranker_best.pt not found, saving current model state")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "reranker_final.pt"))
    print("Saved reranker_final.pt")


if __name__ == "__main__":
    main()
