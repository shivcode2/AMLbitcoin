# 🪙 Bitcoin Anti-Money Laundering with Temporal Graph Neural Networks

This repository implements a **Temporal Graph Neural Network (GNN)** framework for **illicit transaction detection** on the **Elliptic Bitcoin Dataset**.  
The model combines **temporal contrastive pretraining**, **graph attention classification**, and **causal explainability**, offering a **robust and adaptive approach** for financial crime detection in cryptocurrency networks.

---

## 🚀 Overview

This project introduces a **two-stage GNN architecture** designed for anti-money laundering (AML) detection:

### **TemporalGIN Encoder (Pretraining Stage)**
- Learns temporal node embeddings via **contrastive self-supervised learning**.
- Incorporates multi-scale temporal dynamics with GRU-based memory.
- Uses **edge dropout** and **feature noise** for robust augmentations.

### **GAT Classifier (Downstream Stage)**
- Classifies transactions (licit vs illicit) using **attention-based message passing**.
- Employs **focal loss**, **temperature scaling**, and **adaptive thresholds** for handling class imbalance and label drift.

**Additional strategies include:**
- Domain-adaptive importance weighting  
- Pseudo-label consistency learning  
- Self-training on unknown labels  
- Prior-shift correction via EM estimation  
- Temporal drift handling and per-timestep adaptive calibration

---

## 📊 Dataset

**Elliptic Bitcoin Dataset**  
Source: [Kaggle - Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

| File | Description |
|------|--------------|
| `elliptic_txs_features.csv` | Node features (165 transaction-level attributes) |
| `elliptic_txs_edgelist.csv` | Directed edges (transaction flow relationships) |
| `elliptic_txs_classes.csv` | Class labels: `1` (illicit), `2` (licit), `unknown` |

**Temporal split:**
- Train window: `t ≤ 30`  
- Validation: `31 ≤ t ≤ 34`  
- Test: `t > 34`
