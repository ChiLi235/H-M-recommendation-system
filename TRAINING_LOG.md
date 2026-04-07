# Two-Tower Model Training Log

## Baseline Results (before fixes)
```
Two-Tower ONLY      map@12=0.0002  ndcg@12=0.0004  hr@12=0.0018
Two-Tower + SASRec  map@12=0.0000  ndcg@12=0.0001  hr@12=0.0002
```
Reranker was making things *worse* — indicating fundamental bugs in the pipeline.

---

## Bug Fixes Applied (before tuning)

| # | Bug | Fix |
|---|-----|-----|
| 1 | Sequence temporal leakage — full sequence including future purchases was fed into training | Filter sequences to only include purchases strictly before the target week |
| 2 | Article ID encoding conflict — LabelEncoder starts at 0, which is also `padding_idx` | Offset all encoded IDs by +1 to reserve 0 for padding |
| 3 | `user_ever_bought_article` target leakage — computed from the same week's transactions | Compute from temporally filtered sequence (history before target week only) |
| 4 | Parquet array columns loaded as `numpy.ndarray` but checked with `isinstance(x, list)` | Accept both `list` and `np.ndarray` in type checks |
| 5 | Truth-value check `if s_arts and s_dates` on numpy arrays raises `ValueError` | Replace with `len(s_arts) > 0` |
| 6 | Missing cross-features at inference — `user_ever_bought_article` and `user_affinity_prodtype` were always 0 | Compute cross-features inside `_build_reranker_batch()` during inference |
| 7 | `year_week` not passed to reranker during evaluation — weekly sales stats defaulted to 0 | Pass current test week to `rerank_candidates()` |
| 8 | Two-Tower only evaluation used fake scores `range(500, 0, -1)` | Use actual FAISS similarity scores |
| 9 | Only 5 training epochs — likely underfitting | Increased to 15 epochs |

---

## Training Runs & Observations

### Run 1 — Baseline (5 epochs, LR=1e-3, dropout=0.1, temp=0.07)
```
Epoch 1/15  train_loss=6.6420  val_loss=6.9364  ✓ best
Epoch 2/15  train_loss=6.5554  val_loss=6.9740
...
```
**Observation:** val_loss increasing from epoch 2 — overfitting signal.

---

### Run 2 — Increased Regularization (dropout=0.3, AdamW weight_decay=1e-4)
```
Epoch 1/15  train_loss=6.6682  val_loss=6.8963  ✓ best
Epoch 2/15  train_loss=6.5853  val_loss=6.9489
Epoch 3/15  train_loss=6.5504  val_loss=7.0100
...
```
**Changes & Reasoning:**
- `Dropout 0.1 → 0.3`: Dropout randomly zeros out neurons during training, forcing the network to not rely on any single feature. A higher rate (0.3) means more neurons are dropped per forward pass, making the model less likely to memorise training-specific patterns and more likely to learn generalisable representations.
- `Adam → AdamW (weight_decay=1e-4)`: Standard Adam does not properly apply L2 regularisation because it scales the weight penalty by the adaptive learning rate, making it ineffective for large weights. AdamW decouples weight decay from the gradient update, applying a small consistent penalty to all weights every step. This shrinks large weights and prevents the model from over-committing to any particular feature.

**Observation:** val_loss gap slightly reduced but still diverging from epoch 1.

---

### Run 3 — Temperature Fix + Early Stopping (temp=0.1, patience=3)
```
Epoch 1/15  train_loss=6.6650  val_loss=6.8950  ✓ best
Epoch 2/15  train_loss=6.5851  val_loss=6.9488
Epoch 3/15  train_loss=6.5497  val_loss=6.9898
Epoch 4/15  train_loss=6.5257  val_loss=6.9999  → Early stopping
```
**Changes & Reasoning:**
- `Temperature 0.07 → 0.1`: Temperature controls how sharp (confident) the softmax distribution is over in-batch negatives. At 0.07, the model is forced to be extremely confident, which works well at large batch sizes (e.g. 65k in CLIP) where there are many diverse negatives. At batch=1024, a temperature this low makes the loss hypersensitive to small differences in similarity scores, causing the model to overfit to the exact negative items seen in each training batch rather than learning a general embedding space. Raising it to 0.1 softens the distribution and makes training more stable.
- `Early stopping (patience=3)`: Once val_loss stops improving, continuing to train only makes the model overfit further. Early stopping saves the best checkpoint and halts training automatically, avoiding wasted compute and returning the best-generalising weights.

**Observation:** Early stopping working correctly. Best model still from epoch 1 — divergence starts immediately after.

---

### Run 4 — Lower LR + Warmup Scheduler (in progress)
**Changes & Reasoning:**
- `LR 1e-3 → 3e-4`: A learning rate of 1e-3 causes the optimiser to take large parameter steps. In the very first epoch, embeddings are randomly initialised, so large steps push weights into a region that fits the training set well but generalises poorly. 3e-4 is the standard AdamW sweet spot — small enough to navigate the loss landscape carefully, large enough to converge within 15 epochs.
- `Linear warmup (2 epochs) → CosineAnnealingLR`: Starting at the full learning rate from step 0 causes aggressive, destabilising updates while embeddings are still random. Warmup linearly increases the LR from 10% to 100% over the first 2 epochs, letting embeddings settle into a reasonable region before full-strength updates begin. Cosine annealing then smoothly decays the LR for the remaining epochs, avoiding sharp loss spikes near convergence.

---

## Current Config
```python
BATCH_SIZE  = 1024
EPOCHS      = 15
LR          = 3e-4
TEMPERATURE = 0.1
DROPOUT     = 0.3
WEIGHT_DECAY = 1e-4
SCHEDULER   = LinearLR(warmup 2 epochs) → CosineAnnealingLR
EARLY_STOP_PATIENCE = 3
```

---

## Next Steps (if Run 4 still diverges)
- The persistent train/val divergence from epoch 1 may indicate a **temporal distribution shift** between train (Oct 2019–Jun 2020) and val (Jul–Aug 2020) rather than an optimizer issue — fashion trends shift seasonally
- Consider shortening the training window to be closer to the val period
- Investigate whether val users have significantly different purchase patterns than train users
