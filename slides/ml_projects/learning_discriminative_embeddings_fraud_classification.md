# Learning Discriminative Embeddings for Extremely Imbalanced Classification

> A comprehensive guide to transformer-based user behavior modeling for fraud detection with extreme class imbalance (0.001%–1% positive rate)

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Key Challenges](#key-challenges)
- [Architecture Approaches](#architecture-approaches)
  - [Self-Supervised Pre-training](#1-self-supervised-pre-training-on-behavior-sequences)
  - [Contrastive Learning](#2-contrastive-learning-for-extreme-imbalance)
  - [Deep One-Class Classification](#3-deep-one-class-classification)
- [Data Preprocessing](#data-preprocessing)
  - [Action Vocabulary](#action-vocabulary)
  - [Temporal Encoding](#temporal-encoding)
  - [Multi-Field Features](#multi-field-features)
- [Training Objectives](#training-objectives)
- [Evaluation Metrics](#evaluation-metrics)
- [Open Datasets](#open-datasets)
- [Implementation Guide](#implementation-guide)
- [References](#references)

---

## Problem Overview

Fraud detection from behavioral sequences fundamentally differs from recommendation systems. While recommendation focuses on predicting the next likely action, fraud detection requires **learning discriminative embeddings** that separate rare malicious users (0.001%–1%) from the overwhelming majority of legitimate users.

### Key Differences from Recommendation

| Aspect | Recommendation | Fraud Detection |
|--------|---------------|-----------------|
| **Objective** | Predict next action | Classify user/session |
| **Class Balance** | N/A (ranking task) | Extreme imbalance (1:1000 to 1:100000) |
| **Embedding Goal** | Capture preferences | Separate fraud from normal |
| **Evaluation** | HR@K, NDCG@K | AUPRC, Precision@Recall |

---

## Key Challenges

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Extreme Imbalance** | 0.001%–1% positive (fraud) rate | Standard classifiers predict all negatives |
| **Scarce Labels** | Few confirmed fraud cases | Insufficient for supervised learning |
| **Diverse Fraud Patterns** | Fraudsters exhibit varied behaviors | Models overfit to known patterns |
| **Concept Drift** | Fraud tactics evolve over time | Model performance degrades |
| **Representation Collapse** | Contrastive learning fails with binary imbalance | Embeddings don't separate classes |

---

## Architecture Approaches

### 1. Self-Supervised Pre-training on Behavior Sequences

Pre-train on large unlabeled behavioral data, then fine-tune for fraud detection.

#### UB-PTM (User Behavior Pre-Training Model)

> **Paper:** [User Behavior Pre-training for Online Fraud Detection](https://dl.acm.org/doi/10.1145/3534678.3539126) (KDD 2022)

Three self-supervised tasks designed specifically for fraud detection:

| Task | Description | Purpose |
|------|-------------|---------|
| **Hierarchical Action Mask** | Mask individual actions, predict from context | Learn action relationships |
| **Intention Mask** | Mask entire semantic branches of behavior trees | Capture user intentions |
| **Homologous Sequence Prediction** | Predict if two sequences belong to same user | Model user consistency |

**Input Embeddings:**
```
Final Embedding = Action Emb + Action Position Emb + Intention Emb + 
                  Intention Position Emb + Sequence Segment Emb
```

#### UserBERT

> **Paper:** [UserBERT: Pre-training User Model with Contrastive Self-supervision](https://arxiv.org/abs/2109.01274) (SIGIR 2022)

| Component | Description |
|-----------|-------------|
| **Masked Behavior Prediction** | Model relatedness between user behaviors |
| **Behavior Sequence Matching** | Capture consistent interests across time periods |
| **Medium-Hard Negative Sampling** | Select informative negatives for contrastive learning |

---

### 2. Contrastive Learning for Extreme Imbalance

Standard supervised contrastive learning (SupCon) **collapses** with binary imbalanced data.

> **Paper:** [A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets](https://arxiv.org/abs/2503.17024) (2025)

**Problem:** SupCon yields representation spaces dominated by the majority class, with canonical metrics (SAD, CAD) failing to detect collapse.

#### Specialized Frameworks

| Framework | Paper | Key Innovation |
|-----------|-------|----------------|
| **CLeAR** | [Robust Sequence-Based Self-Supervised Representation Learning for Anti-Money Laundering](https://dl.acm.org/doi/10.1145/3627673.3680078) (CIKM 2024) | Intensity-Aware Transformer for extremely low anomaly rates |
| **ConRo** | [Robust Fraud Detection via Supervised Contrastive Learning](https://arxiv.org/abs/2308.10055) (2023) | Data augmentation to generate diverse malicious sessions |
| **ECL** | [Equilibrium Contrastive Learning for Imbalanced Image Classification](https://arxiv.org/abs/2602.09506) (2025) | Geometric equilibrium in representation space |

#### ConRo Framework

```
┌─────────────────────────────────────────────────────────┐
│                    ConRo Pipeline                        │
├─────────────────────────────────────────────────────────┤
│ 1. Data Augmentation → Generate diverse malicious       │
│ 2. Supervised Contrastive Learning → Learn separable    │
│    representations                                       │
│ 3. Open-Set Classification → Detect unseen fraud types  │
└─────────────────────────────────────────────────────────┘
```

---

### 3. Deep One-Class Classification

For extreme imbalance, train only on normal users and detect anomalies.

> **Paper:** [Deep One-Class Classification](https://proceedings.mlr.press/v80/ruff18a.html) (ICML 2018)

#### Deep SVDD (Support Vector Data Description)

**Concept:** Map normal data to a hypersphere; anomalies fall outside the boundary.

```
                    ┌─────────────┐
     Normal Users → │ Transformer │ → Embeddings → Hypersphere Center
                    │   Encoder   │              ↓
                    └─────────────┘         Distance = Anomaly Score
```

**Advantages:**
- Only requires normal data for training
- Avoids class imbalance entirely
- Well-suited for scenarios with abundant normal data and scarce anomalies

> **Extended Paper:** [Deep One-Class Classification Model Assisted by Radius Constraint](https://www.sciencedirect.com/science/article/abs/pii/S095219762401515X) (2024)

---

## Data Preprocessing

### Action Vocabulary

Build a vocabulary of discrete action tokens from behavioral events.

```python
# Example vocabulary structure
vocabulary = {
    # Special tokens
    "[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3, "[UNK]": 4,
    
    # Action tokens (action_type + target)
    "click_checkout_button": 5,
    "view_account_page": 6,
    "submit_login_form": 7,
    "scroll_product_list": 8,
    # ... domain-specific actions
}
```

**Composite Token Strategy (like BST):**
```python
# Combine action + target + context
token_embedding = action_emb + target_emb + page_emb
```

### Temporal Encoding

| Method | Description | Reference |
|--------|-------------|-----------|
| **Relative Time** | `pos(v) = t_current - t_event` | BST (Alibaba) |
| **Inter-Event Gaps** | Bucket time between actions | TiSASRec |
| **Dwell Time** | Duration on each page/action | FraudTransformer |
| **TO-RoPE** | Rotary encoding for both index and wall-clock time | Meta (2025) |

> **Paper:** [FraudTransformer: Time-Aware GPT for Transaction Fraud Detection](https://arxiv.org/abs/2509.23712) (2025)

```python
# Time encoding example
class TimeEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.time_mlp = nn.Linear(1, d_model)
    
    def forward(self, timestamps):
        # Compute inter-event gaps
        gaps = timestamps[1:] - timestamps[:-1]
        gaps = torch.cat([torch.zeros(1), gaps])
        return self.time_mlp(gaps.unsqueeze(-1))
```

### Multi-Field Features

Each event may have multiple attributes: action_type, page, element_id, device, etc.

| Strategy | Method | Pros/Cons |
|----------|--------|-----------|
| **Additive** | Sum field embeddings | Simple, used in BST |
| **Concatenative** | Concat → Linear → d-dim | More expressive, higher compute |
| **Unified Stream** | Convert all features to tokens | Like HSTU, handles any feature type |

---

## Training Objectives

### Comparison for Fraud Detection

| Objective | Signal Density | Best For | Key Paper |
|-----------|---------------|----------|-----------|
| **Masked Action Prediction** | Low (ρ% positions) | Pre-training, learning relationships | BERT4Rec |
| **Next Action Prediction** | High (all positions) | Real-time scoring, dense signal | SASRec |
| **Contrastive (sequence-level)** | Auxiliary | Extreme imbalance, few positives | CL4SRec, ConRo |
| **Deep SVDD** | N/A | Train only on normals | Deep SVDD |
| **Focal Loss** | High | Reduce easy negative dominance | Lin et al. |

### Recommended Two-Stage Approach

```
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Self-Supervised Pre-training (All Users, Unlabeled) │
├──────────────────────────────────────────────────────────────┤
│ • Masked Action Prediction (15-20% mask rate)                │
│ • Sequence Matching (same user = positive)                   │
│ • Optional: Contrastive Learning on augmented sequences      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Fine-tuning for Fraud Detection                     │
├──────────────────────────────────────────────────────────────┤
│ Option A: Deep SVDD on normal user embeddings only           │
│ Option B: Focal Loss + Supervised Contrastive + Oversampling │
└──────────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics

### Why Standard Metrics Fail

| Metric | Problem with Extreme Imbalance |
|--------|-------------------------------|
| **Accuracy** | 99.9% by predicting all negatives |
| **ROC-AUC** | Misleading; doesn't reflect precision at low FPR |
| **F1-Score** | Threshold-dependent, may not reflect operational needs |

### Recommended Metrics

| Metric | Description | When to Use |
|--------|-------------|-------------|
| **AUPRC** | Area Under Precision-Recall Curve | Primary metric for imbalanced data |
| **Precision@Recall=X%** | Precision at fixed recall threshold | Operational decision-making |
| **F1 at Optimal Threshold** | Best F1 across all thresholds | Overall performance summary |
| **False Positive Rate** | Rate of false alarms | User experience impact |

```python
from sklearn.metrics import precision_recall_curve, auc

def evaluate_fraud_detection(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    
    # Precision at 80% recall
    idx = np.argmin(np.abs(recall - 0.80))
    precision_at_80_recall = precision[idx]
    
    return {
        'AUPRC': auprc,
        'Precision@Recall=80%': precision_at_80_recall,
    }
```

---

## Open Datasets

### 1. CERT Insider Threat Dataset ⭐ (Recommended)

> Synthetic insider-threat benchmark with ground-truth labels for anomaly detection

| Version | Users | Duration | Threat Types |
|---------|-------|----------|--------------|
| v4.2 | 1,000 | 501 days | Sabotage, Data Exfiltration |
| v5.2 | 2,000 | 18 months | + Espionage |
| v6.2 | 4,000 | 18 months | + IP Theft |

**Event Types:** Logon/Logoff, Device Usage, File Access, Email, HTTP Activity

| Resource | Link |
|----------|------|
| 📥 Download | [CMU CERT Division](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) |
| 📄 Paper | [Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data](https://ieeexplore.ieee.org/document/6565236) |
| 💻 Code | [github.com/RobertoDure/Insider_Threat_Detection_with_CERT](https://github.com/RobertoDure/Insider_Threat_Detection_with_CERT) |

---

### 2. IEEE-CIS Fraud Detection Dataset

> Real-world e-commerce transaction fraud from Vesta Corporation

| Statistic | Value |
|-----------|-------|
| Transactions | 590,540 |
| Features | 393 (raw), 67 (benchmark) |
| Fraud Rate | 3.5% |

**Features:** Transaction amount, time deltas (D1-D15), card info, device info, Vesta engineered features (V1-V339)

| Resource | Link |
|----------|------|
| 📥 Download | [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) |
| 📄 Winning Solution | [NVIDIA Blog: Winning Kaggle Solution](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/) |
| 💻 Top 5% Solution | [Medium: IEEE-CIS Fraud Detection](https://medium.com/data-science/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f) |

---

### 3. Amazon Fraud Dataset Benchmark (FDB)

> Standardized benchmark covering multiple fraud detection tasks

| Dataset | Task | Fraud Rate | Size |
|---------|------|------------|------|
| IEEE-CIS | Transaction Fraud | 3.5% | 590K |
| Credit Card | Card Fraud | 0.17% | 284K |
| Fake Jobs | Job Posting Fraud | 4.8% | 18K |
| Vehicle Loan | Loan Default | 21.7% | 233K |
| Bot Detection | Malicious Traffic | - | - |

| Resource | Link |
|----------|------|
| 📥 Download | [github.com/amazon-science/fraud-dataset-benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) |
| 📄 Documentation | [FDB README](https://github.com/amazon-science/fraud-dataset-benchmark#readme) |

---

### 4. Renren Clickstream Dataset

> Clickstream traces for Sybil (fake account) detection

| Class | Users | Duration |
|-------|-------|----------|
| Normal | 5,998 | 2 months |
| Sybil | 9,994 | 2 months |

**Data:** Sequence of HTTP requests (clicks, page views, actions)

| Resource | Link |
|----------|------|
| 📄 Paper | [Clickstream User Behavior Models](https://dl.acm.org/doi/10.1145/3068332) (ACM TWEB 2017) |
| 📄 PDF | [people.cs.uchicago.edu](https://people.cs.uchicago.edu/~ravenben/publications/pdf/clickstream-tweb17.pdf) |

---

### 5. Multimodal Banking Dataset (MBD)

> Large-scale multimodal financial dataset

| Modality | Events | Duration |
|----------|--------|----------|
| Money Transfers | 950M | 1-2 years |
| Geo Position | 1B | 1-2 years |
| Support Dialogs | 5M | 1-2 years |
| Product Purchases | - | Monthly |

**Downstream Tasks:** Purchase prediction, client matching, churn prediction

| Resource | Link |
|----------|------|
| 📄 Paper | [Multimodal Banking Dataset](https://arxiv.org/abs/2409.17587) |
| 📥 Related: Data Fusion | [ods.ai/tracks/data-fusion-2022](https://ods.ai/tracks/data-fusion-2022-competitions) |
| 📥 Related: Alfabattle | [boosters.pro/championship/alfabattle2](https://boosters.pro/championship/alfabattle2/overview) |

---

### 6. RecBole Sequential Datasets

> Standard benchmarks for sequence modeling (for pre-training)

| Dataset | Users | Items | Interactions | Download |
|---------|-------|-------|--------------|----------|
| Amazon-Beauty | 22K | 12K | 198K | [RecBole](https://recbole.io/dataset_list.html) |
| Amazon-Electronics | 192K | 63K | 1.6M | [RecBole](https://recbole.io/dataset_list.html) |
| MovieLens-1M | 6K | 4K | 1M | [RecBole](https://recbole.io/dataset_list.html) |
| Yelp | 31K | 38K | 1.5M | [RecBole](https://recbole.io/dataset_list.html) |
| Gowalla | 107K | 1.3M | 6.4M | [RecBole](https://recbole.io/dataset_list.html) |

---

## Implementation Guide

### Recommended Architecture

```python
import torch
import torch.nn as nn

class BehaviorTransformer(nn.Module):
    """Transformer encoder for user behavior sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Embeddings
        self.action_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.time_encoder = nn.Linear(1, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, action_ids, timestamps, attention_mask=None):
        seq_len = action_ids.size(1)
        positions = torch.arange(seq_len, device=action_ids.device)
        
        # Combine embeddings
        x = self.action_embedding(action_ids)
        x = x + self.position_embedding(positions)
        x = x + self.time_encoder(timestamps.unsqueeze(-1))
        
        # Transformer encoding
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        return self.layer_norm(x)
    
    def get_user_embedding(self, action_ids, timestamps, attention_mask=None):
        """Extract fixed-length user embedding via mean pooling."""
        hidden_states = self.forward(action_ids, timestamps, attention_mask)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(1) / mask.sum(1)
        else:
            pooled = hidden_states.mean(dim=1)
        
        return pooled
```

### Training Pipeline

```python
# Stage 1: Self-supervised pre-training
pretrain_objectives = [
    MaskedActionPrediction(mask_prob=0.15),
    SequenceMatching(temperature=0.07),
]

for epoch in range(pretrain_epochs):
    for batch in pretrain_loader:
        loss = sum(obj(model, batch) for obj in pretrain_objectives)
        loss.backward()
        optimizer.step()

# Stage 2: Fine-tune for fraud detection
# Option A: Deep SVDD
svdd = DeepSVDD(model, center=compute_center(normal_users))
for epoch in range(finetune_epochs):
    for batch in normal_user_loader:
        embeddings = model.get_user_embedding(**batch)
        loss = svdd.hypersphere_loss(embeddings)
        loss.backward()
        optimizer.step()

# Option B: Supervised with focal loss
classifier = nn.Linear(d_model, 1)
focal_loss = FocalLoss(gamma=2.0, alpha=0.99)  # High alpha for imbalance

for epoch in range(finetune_epochs):
    for batch in labeled_loader:
        embeddings = model.get_user_embedding(**batch)
        logits = classifier(embeddings)
        loss = focal_loss(logits, batch['labels'])
        loss.backward()
        optimizer.step()
```

---

## References

### Core Papers

| Paper | Venue | Topic | Link |
|-------|-------|-------|------|
| User Behavior Pre-training for Online Fraud Detection | KDD 2022 | Pre-training for fraud | [ACM](https://dl.acm.org/doi/10.1145/3534678.3539126) |
| UserBERT: Contrastive User Model Pre-training | SIGIR 2022 | Contrastive pre-training | [arXiv](https://arxiv.org/abs/2109.01274) |
| Deep One-Class Classification | ICML 2018 | Deep SVDD | [PMLR](https://proceedings.mlr.press/v80/ruff18a.html) |
| Robust Fraud Detection via Supervised Contrastive Learning | arXiv 2023 | ConRo framework | [arXiv](https://arxiv.org/abs/2308.10055) |
| CLeAR: Anti-Money Laundering | CIKM 2024 | Intensity-aware transformer | [ACM](https://dl.acm.org/doi/10.1145/3627673.3680078) |
| FraudTransformer | arXiv 2025 | Time-aware GPT for fraud | [arXiv](https://arxiv.org/abs/2509.23712) |

### Transformer Architectures for Sequences

| Paper | Venue | Topic | Link |
|-------|-------|-------|------|
| SASRec | ICDM 2018 | Self-attentive sequential rec | [arXiv](https://arxiv.org/abs/1808.09781) |
| BERT4Rec | CIKM 2019 | Bidirectional sequential rec | [arXiv](https://arxiv.org/abs/1904.06690) |
| BST (Behavior Sequence Transformer) | KDD 2019 | Alibaba production system | [arXiv](https://arxiv.org/abs/1905.06874) |
| HSTU (Meta) | ICML 2024 | Trillion-parameter generative rec | [arXiv](https://arxiv.org/abs/2402.17152) |
| Transformers4Rec | RecSys 2021 | NVIDIA sequential rec library | [ACM](https://dl.acm.org/doi/10.1145/3460231.3474255) |

### Imbalanced Learning

| Paper | Venue | Topic | Link |
|-------|-------|-------|------|
| A Tale of Two Classes | arXiv 2025 | SupCon for binary imbalance | [arXiv](https://arxiv.org/abs/2503.17024) |
| Equilibrium Contrastive Learning | arXiv 2025 | Geometric equilibrium | [arXiv](https://arxiv.org/abs/2602.09506) |
| Focal Loss | ICCV 2017 | Dense object detection | [arXiv](https://arxiv.org/abs/1708.02002) |

### Insider Threat & Anomaly Detection

| Paper | Venue | Topic | Link |
|-------|-------|-------|------|
| UBS-Transformer for Insider Threat | arXiv 2025 | CERT dataset + Transformer | [arXiv](https://arxiv.org/abs/2506.23446) |
| Insider Threat Detection via User Behavior | Applied Sciences 2019 | CERT modeling | [MDPI](https://www.mdpi.com/2076-3417/9/19/4018) |
| One-Class Classification Survey | J Big Data 2021 | OCC methods review | [Springer](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00514-x) |

---

## Code Repositories

| Repository | Description | Link |
|------------|-------------|------|
| **amazon-science/fraud-dataset-benchmark** | Standardized fraud detection benchmark | [GitHub](https://github.com/amazon-science/fraud-dataset-benchmark) |
| **Insider_Threat_Detection_with_CERT** | Full pipeline for CERT dataset | [GitHub](https://github.com/RobertoDure/Insider_Threat_Detection_with_CERT) |
| **Fraud_Detection_with_Sequential_User_Behavior** | Transformer for clickstream fraud | [GitHub](https://github.com/avayang/Fraud_Detection_with_Sequential_User_Behavior) |
| **RecBole** | Unified recommendation library (94+ algorithms) | [GitHub](https://github.com/RUCAIBox/RecBole) |
| **NVIDIA-Merlin/Transformers4Rec** | Sequential recommendation with HuggingFace | [GitHub](https://github.com/NVIDIA-Merlin/Transformers4Rec) |
| **safe-graph/graph-fraud-detection-papers** | Curated list of fraud detection papers | [GitHub](https://github.com/safe-graph/graph-fraud-detection-papers) |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-repo/fraud-detection-embeddings.git
cd fraud-detection-embeddings
pip install -r requirements.txt

# Download CERT dataset
python scripts/download_cert.py --version 4.2

# Preprocess into sequences
python scripts/preprocess_cert.py --output data/processed/

# Pre-train transformer
python train.py --stage pretrain --config configs/pretrain.yaml

# Fine-tune for fraud detection
python train.py --stage finetune --config configs/finetune_svdd.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best.pt --metrics auprc,precision_at_recall
```
