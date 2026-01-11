# PROJECT PLAN: Movie Recommender dengan Neural Collaborative Filtering

## 1. Pendahuluan

### 1.1 Latar Belakang
Sistem rekomendasi film menggunakan Deep Learning untuk memprediksi preferensi pengguna. Project ini mengimplementasikan **Neural Collaborative Filtering (NCF)** dengan variasi **NeuMF + Genre & Synopsis + Gated Feature Fusion** sebagai model usulan.

### 1.2 Tujuan
1. Membangun pipeline data lengkap (ratings, movies, links, metadata)
2. Mengimplementasikan baseline: GMF, MLP, NeuMF
3. Mengembangkan model NCF dengan fitur genre & sinopsis
4. Membandingkan performa semua model melalui ablation study
5. Membangun pipeline inference untuk rekomendasi top-K

### 1.3 Relevansi dengan PPT
| PPT Requirement | Implementation |
|-----------------|----------------|
| Load ratings, movies, links, metadata | `data_loader.py` |
| Join movies-links-metadata via tmdbId | Preprocessing step |
| Hapus data tidak lengkap/duplikasi | Cleaning step |
| Filter sparse users/movies | Filtering step |
| Time-based split (train/val/test) | `train_test_split()` |
| GMF, MLP, NeuMF baselines | 3 model files |
| NeuMF + genre + sinopsis + gated fusion | `neumf_plus.py` |

---

## 2. Dataset Overview

### 2.1 File yang Tersedia
```
datasets/
├── ratings.csv      (~709MB, ~25M ratings)
├── movies_metadata.csv  (~34MB, ~45K movies)
├── links.csv        (~989KB, ~45K links)
└── keywords.csv     (~6MB)
```

### 2.2 Schema
| File | Kolom |
|------|-------|
| `ratings.csv` | userId, movieId, rating, timestamp |
| `movies_metadata.csv` | title, overview, genres, id (tmdbId), ... |
| `links.csv` | movieId, imdbId, tmdbId |
| `keywords.csv` | id, keywords (optional) |

### 2.3 Estimasi Setelah Preprocessing
| Metric | Estimasi |
|--------|----------|
| Total users | ~270K |
| Total movies | ~45K |
| Total ratings | ~25M |
| Movies dengan synopsis | ~40K |

---

## 3. Arsitektur Model

### 3.1 Baseline Models

#### A. GMF (Generalized Matrix Factorization)
```
User ID → Embedding(32) ─┐
                        ├─→ Element-wise Multiply → FC(8) → Output
Item ID → Embedding(32) ─┘
```

#### B. MLP (Multi-Layer Perceptron)
```
User ID → Embedding(32) ─┐
                        ├─→ Concat → FC(128) → FC(64) → Dropout → FC(32) → Output
Item ID → Embedding(32) ─┘
```

#### C. NeuMF (Neural Matrix Factorization)
```
GMF branch ─┐
            ├─→ Concat → FC(32) → Output
MLP branch ─┘
```

### 3.2 Model Usulan: NeuMF + Genre + Synopsis + Gated Fusion

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER & ITEM EMBEDDINGS                      │
├─────────────────────────────────────────────────────────────────┤
│  User ID → Embed(32) ─┐                                          │
│  Item ID → Embed(32) ─┤                                          │
│                       ├──→ GMF (elem-mult) ──┐                   │
│                                             │                   │
│                       └──→ MLP ──────────────┼──→ NeuMF ──┐     │
│                                             │            │     │
├─────────────────────────────────────────────┤            │     │
│                      CONTENT FEATURES       │            │     │
├─────────────────────────────────────────────┤            │     │
│  Genres → Multi-hot(19) ──→ FC(64) ────────┼────┐       │     │
│  Synopsis → Sentence-BERT ──→ Embed(384) ──┘    │       │     │
│                                                 │       │     │
├─────────────────────────────────────────────────┴───────┘     │
│                    GATED FEATURE FUSION                       │
│  neumf_embed + content_embed → Gate Network → Weighted Fusion│
├─────────────────────────────────────────────────────────────────┤
│                      OUTPUT LAYER                               │
│  FC(64) → Dropout(0.2) → FC(1) → Sigmoid                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Gated Feature Fusion Module

```python
class GatedFusion(nn.Module):
    """
    Dynamically balances collaborative filtering and content signals.
    Gate value close to 1 → rely more on CF embeddings
    Gate value close to 0 → rely more on content features
    """
    def __init__(self, cf_dim, content_dim, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cf_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, cf_embed, content_embed):
        combined = torch.cat([cf_embed, content_embed], dim=-1)
        gate = self.gate(combined)  # (batch, 1)
        fused = gate * cf_embed + (1 - gate) * content_embed
        return fused, gate
```

### 3.4 Content Encoding Strategy

**Decision: Use Pre-trained Sentence-BERT instead of TextCNN**

| Aspect | TextCNN | Sentence-BERT |
|--------|---------|---------------|
| Training | From scratch | Pre-trained |
| Data needed | Large corpus | Works with small data |
| Semantic understanding | Local patterns | Full sentence meaning |
| Setup complexity | High (tokenization, vocab) | Low (pip install) |
| Performance | Moderate | Better |

```python
# Synopsis encoding using sentence-transformers
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, fast
synopsis_embed = embedder.encode(synopsis_text)  # (384,)
```

---

## 4. Metodologi

### 4.1 Preprocessing Pipeline

```
Step 1: Load Data
├── Load ratings.csv
├── Load movies_metadata.csv
├── Load links.csv
└── Load keywords.csv (optional)

Step 2: Data Merging
├── Join links ↔ movies_metadata via tmdbId
├── Join ratings ↔ merged data via movieId
└── Extract genres dari JSON string

Step 3: Data Cleaning & Quality Checks
├── Verify rating range: 0.5 - 5.0
├── Remove rows dengan overview = empty/NaN
├── Remove duplicates (userId, movieId)
├── Remove movies tanpa genre
├── Convert timestamp ke datetime
├── Verify timestamp ordering (monotonic increasing)
└── Check: all movies in ratings exist in metadata

Step 4: Filtering
├── Filter users dengan < 5 ratings
├── Filter movies dengan < 5 ratings
└── Log: statistics before/after filtering

Step 5: Train/Val/Test Split (Time-based)
├── Sort by timestamp PER USER
├── Train: 70% (terawal)
├── Val: 15% (tengah)
└── Test: 15% (terakhir)

Step 6: Create Cold-Start Evaluation Set
├── Cold-start users: users with ≤ 10 ratings in train
├── Cold-start items: items with ≤ 10 ratings in train
└── This is where content features should shine

Step 7: Save Processed Data
├── train.pkl, val.pkl, test.pkl
├── cold_start_test.pkl
├── user_map.pkl, item_map.pkl
├── genre_encoder.pkl
└── data_statistics.json
```

### 4.2 Negative Sampling Strategy

**CRITICAL: Dynamic Negative Sampling During Training**

```python
class NegativeSampler:
    """
    Sample negatives DURING training, not pre-sampled.
    This ensures diversity and better convergence.
    """
    def __init__(self, num_negatives=4, sampling_strategy='uniform'):
        self.num_negatives = num_negatives
        self.strategy = sampling_strategy  # 'uniform' or 'popularity'

    def sample(self, users, pos_items, all_items):
        """
        For each (user, pos_item) pair, sample N negative items.
        Ensures negatives are NOT in user's interaction history.
        """
        neg_items = []
        for user, pos in zip(users, pos_items):
            # Get items user has NOT interacted with
            user_items = self.user_history[user]
            candidates = np.setdiff1d(all_items, user_items)

            if self.strategy == 'popularity':
                # Sample inversely proportional to popularity
                weights = 1 / (self.item_popularity[candidates] + 1e-8)
                weights = weights / weights.sum()
                neg = np.random.choice(candidates, self.num_negatives,
                                      p=weights, replace=True)
            else:  # uniform
                neg = np.random.choice(candidates, self.num_negatives,
                                      replace=True)
            neg_items.append(neg)
        return np.array(neg_items)
```

### 4.3 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 256 | Adjust based on GPU memory |
| Learning rate | 0.001 | Use ReduceLROnPlateau |
| Optimizer | Adam | weight_decay=1e-5 |
| Negative samples | 4 per positive | Sampled dynamically |
| Epochs | 30 | With early stopping |
| Loss function | BPR Loss | -log(sigma(pos - neg)) |
| Early stopping | patience=5 | On validation HR@10 |
| Gradient clipping | max_norm=5.0 | Prevent exploding gradients |
| Dropout | 0.1-0.2 | In MLP and fusion layers |

### 4.4 Regularization Strategy

```python
# In model forward pass
class NeuMFPlus(nn.Module):
    def __init__(self, ...):
        # ... embeddings and layers ...

        # Dropout for regularization
        self.dropout_fc = nn.Dropout(0.2)
        self.dropout_fusion = nn.Dropout(0.1)

    def forward(self, user_id, item_id, genre_features, synopsis_embed):
        # ... forward pass with dropout ...
        x = self.dropout_fc(x)
        return x
```

### 4.5 Evaluation Metrics

#### Standard Evaluation
| Metric | Deskripsi | K Values |
|--------|-----------|----------|
| Hit Rate @K | % user dengan minimal 1 hit di top-K | 5, 10, 20 |
| NDCG @K | Normalized Discounted Cumulative Gain | 5, 10, 20 |
| AUC | Area Under ROC Curve | - |

#### Cold-Start Evaluation (KEY for content features)
| Subset | Description |
|--------|-------------|
| Cold-Start Users | Users with ≤ 10 interactions in training |
| Cold-Start Items | Items with ≤ 10 interactions in training |

**Why this matters:** Content features (genre + synopsis) should help significantly for cold-start cases where CF has insufficient data.

### 4.6 Ablation Study Plan

```
Progressively add components to measure each contribution:

┌────────────────────────────────────────────────────────────┐
│ Model                      │ Components                     │
├────────────────────────────────────────────────────────────┤
│ 1. GMF                      │ Baseline MF only              │
│ 2. MLP                      │ Baseline MLP only             │
│ 3. NeuMF                    │ GMF + MLP combined            │
│ 4. NeuMF + Genre            │ + Genre features (no gate)    │
│ 5. NeuMF + Synopsis         │ + Synopsis (no gate)          │
│ 6. NeuMF + Genre + Synopsis │ + Both (concat, no gate)      │
│ 7. NeuMF + Genre + Synopsis + Gated Fusion │ FINAL MODEL   │
└────────────────────────────────────────────────────────────┘

Expected improvement pattern:
GMF < MLP < NeuMF < NeuMF+Genre < NeuMF+Both < NeuMF+Gated
     (cold-start improvement should be visible at step 4+)
```

---

## 5. Struktur File & Folder

```
NCF-Movie-Recommender/
│
├── datasets/                    # Data mentah (read-only)
│   ├── ratings.csv
│   ├── movies_metadata.csv
│   ├── links.csv
│   └── keywords.csv
│
├── data/                        # Data hasil preprocessing
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   ├── cold_start_test.pkl
│   ├── mappings.pkl
│   └── statistics.json
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_negative_sampling.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_neumf_implementation.ipynb
│   ├── 06_content_features.ipynb
│   ├── 07_gated_fusion.ipynb
│   ├── 08_ablation_study.ipynb
│   └── 09_evaluation_analysis.ipynb
│
├── src/                         # Python modules
│   ├── __init__.py
│   ├── config.py               # Hyperparameters
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Preprocessing functions
│   ├── negative_sampling.py    # Negative sampling logic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # Base model class
│   │   ├── gmf.py              # GMF model
│   │   ├── mlp.py              # MLP model
│   │   ├── neumf.py            # NeuMF model
│   │   ├── content_encoder.py  # Genre + Synopsis encoding
│   │   └── neumf_plus.py       # NeuMF + Gated Fusion
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation functions
│   ├── inference.py            # Inference pipeline
│   └── utils.py                # Helper functions
│
├── experiments/                 # Experiment results
│   ├── ablation_results.json
│   ├── trained_models/
│   └── logs/
│
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
└── README.md                    # Project documentation
```

---

## 6. Progres & Timeline

| Fase | Aktivitas | Estimasi | Status |
|------|-----------|----------|--------|
| **Phase 1: Setup & Data** |
| 1 | Setup environment & dependencies | 1 jam | ⏳ |
| 2 | Data exploration & statistics | 2 jam | ⏳ |
| 3 | Preprocessing pipeline + quality checks | 3 jam | ⏳ |
| 4 | Create cold-start evaluation sets | 1 jam | ⏳ |
| **Phase 2: Baselines** |
| 5 | Implement GMF baseline | 2 jam | ⏳ |
| 6 | Implement MLP baseline | 2 jam | ⏳ |
| 7 | Implement NeuMF baseline | 2 jam | ⏳ |
| **Phase 3: Content Features** |
| 8 | Implement content encoder (genre + SBERT) | 2 jam | ⏳ |
| 9 | Implement gated fusion module | 2 jam | ⏳ |
| 10 | Integrate: NeuMF + Genre + Synopsis + Gated | 2 jam | ⏳ |
| **Phase 4: Training & Evaluation** |
| 11 | Implement negative sampling | 1 jam | ⏳ |
| 12 | Training with early stopping | 3 jam | ⏳ |
| 13 | Ablation study execution | 2 jam | ⏳ |
| 14 | Cold-start evaluation | 2 jam | ⏳ |
| **Phase 5: Inference & Analysis** |
| 15 | Build inference pipeline | 2 jam | ⏳ |
| 16 | Results analysis & visualization | 2 jam | ⏳ |
| 17 | Documentation & reporting | 2 jam | ⏳ |
| | **TOTAL** | **~33 jam** | |

---

## 7. Dependencies

```txt
# requirements.txt
# Core ML/DL
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Data Processing
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress & Utilities
tqdm>=4.65.0
tensorboard>=2.13.0

# Text Processing (for synopsis)
sentence-transformers>=2.2.0

# Optional: For logging experiments
# wandb>=0.15.0
# mlflow>=2.7.0
```

---

## 8. Implementation Checklist

### Before Training
- [ ] Verify all data files are accessible
- [ ] Run data quality checks and review statistics
- [ ] Create cold-start evaluation sets
- [ ] Implement negative sampling with correctness tests

### During Development
- [ ] Start with ONE baseline (GMF), get it working end-to-end
- [ ] Add evaluation pipeline early (not after all models)
- [ ] Implement ablation study as you go (not at the end)
- [ ] Save checkpoints after each epoch

### During Training
- [ ] Use early stopping (don't guess epoch count)
- [ ] Monitor gradient norms (detect issues early)
- [ ] Log metrics to TensorBoard
- [ ] Save best model based on validation HR@10

### After Training
- [ ] Run ablation study on ALL model variants
- [ ] Evaluate on cold-start test set separately
- [ ] Generate comparison plots
- [ ] Build inference function for top-K recommendations

---

## 9. Critical Implementation Notes

### 9.1 Avoiding Common Pitfalls

```python
# DON'T: Pre-sample negatives once and reuse
negatives = pre_sample_negatives()  # Wrong!

# DO: Sample negatives dynamically during training
for batch in dataloader:
    negs = negative_sampler.sample(batch.users, batch.items)
    # ... train on (user, pos_item, neg_items)

# DON'T: Use global timestamp split
train, test = split_by_timestamp(all_data)  # Leakage!

# DO: Split per-user by timestamp
for user in users:
    user_data = sort_by_timestamp(user_data)
    train, test = split(user_data, [0.7, 0.15, 0.15])

# DON'T: Forget to exclude positives from negatives
neg_items = random.sample(all_items)  # Might include actual positives!

# DO: Ensure negatives are not in user history
candidates = set(all_items) - set(user_positive_items)
```

### 9.2 Model Validation During Training

```python
# Validate model is learning (not memorizing)
def sanity_check(model, train_loader, val_loader):
    # Training loss should decrease
    # Validation metrics should improve initially
    # If both improve immediately → model is working
    # If val metrics never improve → check data/implementation

    train_loss = train_epoch(model, train_loader)
    val_hr = evaluate(model, val_loader, K=10)

    print(f"Train Loss: {train_loss:.4f}, Val HR@10: {val_hr:.4f}")

    # Sanity: HR@10 should be > random (10/num_items)
    # For 45K items, random HR@10 ≈ 10/45000 ≈ 0.0002
    assert val_hr > 0.001, "Model not learning better than random!"
```

### 9.3 Inference Pipeline

```python
def recommend_top_k(model, user_id, k=10, exclude_seen=True):
    """
    Generate top-K recommendations for a user.
    """
    model.eval()
    with torch.no_grad():
        # Get all items
        all_items = torch.arange(num_items, device=device)

        # Get user embeddings
        user_embed = model.user_embedding(user_id)

        # Score all items
        scores = model.predict(user_id, all_items)

        # Optionally exclude seen items
        if exclude_seen:
            seen_items = get_user_history(user_id)
            scores[seen_items] = -float('inf')

        # Get top-K
        top_k_indices = torch.topk(scores, k).indices
        top_k_scores = scores[top_k_indices]

        return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()
```

---

## 10. Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Text encoding | Sentence-BERT | Pre-trained, better semantics, easier setup |
| Negative sampling | Dynamic, per-batch | Better convergence than pre-sampled |
| Split strategy | Per-user time-based | Prevents data leakage |
| Evaluation focus | Cold-start subsets | Validates content feature contribution |
| Experiment tracking | TensorBoard | Simple, no external service needed |
| Regularization | Dropout + weight decay | Proven technique for NCF |

---

## 11. Expected Outcomes

### Success Criteria
1. **Baseline Performance**: NeuMF achieves HR@10 > 0.1 on test set
2. **Content Feature Impact**: +5-10% improvement on cold-start users/items
3. **Gated Fusion Benefit**: +2-5% over simple concatenation
4. **Training Stability**: Converges in < 20 epochs with early stopping

### Red Flags (If these happen, debug immediately)
- HR@10 < 0.01 → Model not learning, check data/implementation
- Training loss decreases but val metrics stay flat → Overfitting, add regularization
- Cold-start performance worse than normal → Content features not helping
- Gate values always > 0.9 or < 0.1 → Gate not learning, check initialization

---

## 12. Referensi

1. He, X., et al. (2017). "Neural Collaborative Filtering." WWW.
2. D2L.ai Chapter 9.6: Neural Collaborative Filtering
3. https://github.com/hexiangnan/neural_collaborative_filtering
4. Sentence-Transformers: https://www.sbert.net/
5. BPR Loss: Rendle, S. (2009). "BPR: Bayesian Personalized Ranking."

---

**Catatan**: Plan ini dirancang untuk menghindari common pitfalls. Fokus pada ablation study dan cold-start evaluation akan membuktikan kontribusi nyata dari content features.
