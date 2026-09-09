# 🛡️ QuantumGuard — Quantum-Enhanced Credit Card Fraud Detection

> **Qiskit Fall Fest 2026 | Centurion University of Technology and Management, Vizianagaram**
> **Hackathon Day 2 · Use Case 01: Credit Card Fraud Detection**

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-hackathon%20prototype-blueviolet">
  <img alt="qiskit" src="https://img.shields.io/badge/built%20with-Qiskit-6929c4">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## 📌 Problem Statement

Fraudulent transactions make up a tiny fraction of all credit card transactions, creating **severe class imbalance**. Classical ML models trained on this kind of data tend to either:

- **Over-flag** legitimate transactions (high false positives → poor customer experience), or
- **Miss real fraud** (high false negatives → direct financial loss).

Banks, card networks, and payment processors need a detection system that stays accurate even when fraud is extremely rare in the data.

---

## 💡 Our Solution

**QuantumGuard** is a hybrid quantum-classical fraud detection pipeline that uses **Quantum Restricted Boltzmann Machines (QRBMs)** and **quantum energy-based models** to learn the underlying structure of transaction data — including the rare, subtle patterns that separate fraud from normal spending behavior — far better than a purely classical model does on imbalanced data.

Instead of only learning a decision boundary (like a classical classifier), our model learns the **energy landscape** of transactions: legitimate transactions settle into low-energy, high-probability regions, while fraudulent ones stand out as high-energy anomalies — even when they're rare.

### Key Idea
```
Classical models  →  learn "is this fraud?" (boundary-based)
Quantum energy models → learn "how normal does this look?" (density-based)
```
This reframing is naturally more robust to class imbalance, because it doesn't need thousands of fraud examples to learn what fraud *isn't*.

---

## 🧭 Methodology

### 1. Data Preparation
- Use a public, anonymized transaction dataset (e.g., the Kaggle *Credit Card Fraud Detection* dataset — PCA-transformed features `V1–V28`, `Time`, `Amount`, `Class`).
- Handle imbalance with a mix of:
  - **Undersampling** of the majority (legitimate) class for quantum-feasible batch sizes.
  - **SMOTE-lite** oversampling on the classical side before quantum encoding.
- Normalize/scale all features to `[0, π]` or `[-1, 1]` so they map cleanly onto qubit rotation angles.

### 2. Dimensionality Reduction (Classical Pre-processing)
- Apply **PCA / Autoencoder compression** to reduce ~30 features down to a small number of components (typically 4–8) that fit within near-term qubit budgets (NISQ constraints).

### 3. Quantum Feature Encoding
- Encode compressed features into quantum states using **angle encoding** or **amplitude encoding** via Qiskit `QuantumCircuit` parameterized rotation gates (`RY`, `RZ`).
- Build a **feature map** (e.g., `ZZFeatureMap` from Qiskit) to capture non-linear correlations between transaction features in Hilbert space.

### 4. Model: Quantum Restricted Boltzmann Machine / Quantum Energy-Based Model
- Visible layer = encoded transaction features; hidden layer = latent qubits capturing correlations.
- Train using a **quantum-classical hybrid loop**:
  1. Quantum circuit estimates the energy of a given transaction state.
  2. Classical optimizer (e.g., COBYLA / SPSA / Adam via Qiskit's `EstimatorQNN` or `SamplerQNN`) updates circuit parameters to minimize reconstruction/contrastive divergence loss.
  3. Repeat until the energy landscape separates "normal" vs "anomalous" transactions.

### 5. Anomaly Scoring & Classification
- For each transaction, compute an **energy score**.
- Transactions above a calibrated energy threshold are flagged as potential fraud.
- Threshold is tuned on a validation split to balance **precision vs recall** (favoring recall, since missed fraud is costlier than a false alarm).

### 6. Evaluation
- Metrics: **Precision, Recall, F1-score, AUPRC** (more meaningful than accuracy on imbalanced data), and **False Negative Rate**.
- Compare against classical baselines: Logistic Regression, Random Forest, XGBoost, Isolation Forest.

---
## 📁 Repository Structure

```
fraud-detection-qml/
├── README.md
├── data/                    ← Hasini
│   └── preprocessing.ipynb
├── classical_baseline/      ← Monisha
│   └── baseline_model.ipynb
├── quantum/
│   ├── feature_map.ipynb    ← Venu
│   └── qsvm_model.ipynb     ← Pratap
├── evaluation/              ← Triveda
│   └── comparison.ipynb
└── results/                 ← everyone dumps final metrics/plots here

```

## 🏗️ Architecture

```
 Raw Transactions
        │
        ▼
 Preprocessing (scaling, imbalance handling)
        │
        ▼
 Classical Dimensionality Reduction (PCA)
        │
        ▼
 Quantum Feature Encoding (ZZFeatureMap)
        │
        ▼
 Quantum RBM / Energy-Based Model (Qiskit EstimatorQNN)
        │
        ▼
 Hybrid Classical-Quantum Training Loop (SPSA/COBYLA optimizer)
        │
        ▼
 Anomaly / Energy Score → Threshold → Fraud / Legitimate
```

---

## 🎯 Expected Outputs

- Fewer false flags on legitimate transactions.
- Fewer missed fraud cases compared to classical baselines.
- Demonstrated **near-zero false negatives** in NQCC (Noisy Quantum Circuit) trial simulations.
- Competitive precision/recall trade-off despite extreme class imbalance.

---

## 🏦 Applies To

- Banks
- Card networks
- Payment processors

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Quantum Framework | Qiskit, Qiskit Machine Learning |
| Classical ML | scikit-learn, imbalanced-learn (SMOTE) |
| Data Handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Optimization | SPSA / COBYLA (Qiskit optimizers) |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone <your-repo-url>
cd quantumguard

# Install dependencies
pip install qiskit qiskit-machine-learning scikit-learn imbalanced-learn pandas numpy matplotlib

# Run the notebook / script
python train_qrbm.py
```

---

## 📈 Future Scope

- Deploy on real quantum hardware (IBM Quantum) instead of simulators for real-world noise benchmarking.
- Extend to real-time streaming transaction scoring.
- Explore Quantum GANs for synthetic fraud sample generation to further address imbalance.

---

## 👥 Team

| Name | Role |
|---|---|
| _Hasini_ | _Data & Preprocessing_ |
| _Monisha_ | _Classical Baseline_ |
| _Venu_ | _Quantum Feature Map_ |
| _Pratap_ | _Quantum Kernel + SVM_ |
| _Triveda_ | _Evaluation & Comparison_ |
| _Nihas_ | _Git & Docs_ |

---

## 📄 License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center"><i>Built for Qiskit Fall Fest 2026 — Centurion University of Technology and Management, Vizianagaram</i></p>
