"""
Quantum Kernel Classifier — Credit Card Fraud Detection
Owners: Venu (feature map) + Pratap (kernel/SVM)

Approach: encode transactions into a quantum feature map (ZZFeatureMap),
compute a quantum kernel (fidelity-based), feed the kernel matrix into
a classical SVM. This is compared against Monisha's classical Random
Forest baseline.

IMPORTANT: quantum simulators/hardware are slow. Do NOT feed this the
full 284k row dataset. Use a small balanced subset (a few hundred rows)
and a small number of PCA components (4-8) = qubits.

Install:
    pip install qiskit qiskit-machine-learning qiskit-ibm-runtime scikit-learn pandas --break-system-packages

IBM backend setup: see ibm_backend.py in the repo root for the one-time
save_account() step. This script imports that module instead of relying
on the default local statevector simulator.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# --- NEW: IBM backend helper (see ibm_backend.py at repo root) ---
from ibm_backend import get_service, pick_backend, build_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_PATH = PROJECT_ROOT / "quantum" / "quantum_kernel_results.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Download creditcard.csv into the data/ folder first.")

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

print(f"Total fraud cases: {len(fraud)}, total legit cases: {len(legit)}")

# ---------------------------------------------------------
# 2. Build a SMALL balanced subset (real hardware/sim can't handle 280k rows)
# ---------------------------------------------------------
# NOTE: on real IBM hardware, keep this even smaller than you would for a
# local simulator run — every extra sample adds queued jobs, not just
# local compute time. Start with 20-40 per class, not 100.
N_PER_CLASS = 30

fraud_sample = resample(fraud, n_samples=min(N_PER_CLASS, len(fraud)),
                         random_state=42, replace=False)
legit_sample = resample(legit, n_samples=N_PER_CLASS,
                         random_state=42, replace=False)

data = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)

X = data.drop('Class', axis=1)
y = data['Class'].values

print(f"Balanced subset shape: {X.shape}, fraud count: {y.sum()}")

# ---------------------------------------------------------
# 3. Reduce dimensions -> number of qubits (keep this small: 4-6)
# ---------------------------------------------------------
N_QUBITS = 4

pca = PCA(n_components=N_QUBITS, random_state=42)
X_reduced = pca.fit_transform(X)

# Quantum feature maps expect inputs roughly in [0, 2*pi] or [-1, 1]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_reduced)

print(f"Reduced to {N_QUBITS} features (= {N_QUBITS} qubits)")

# ---------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------
# 5. Build the quantum feature map (Venu's part) — UNCHANGED
# ---------------------------------------------------------
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2, entanglement='linear')
print("\nFeature map circuit:")
print(feature_map.decompose())

# ---------------------------------------------------------
# 6. Build the quantum kernel + train SVM (Pratap's part) — CHANGED
# ---------------------------------------------------------
# OLD (local statevector simulator, no IBM connection):
#     quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
#
# NEW: connect to a real IBM backend via Qiskit Runtime.
print("\nConnecting to IBM Quantum...")
service = get_service()
backend = pick_backend(service, min_qubits=N_QUBITS)
print(f"Using backend: {backend.name} (queue: {backend.status().pending_jobs} pending jobs)")

quantum_kernel, pass_manager, sampler = build_kernel(feature_map, backend)

print("\nComputing quantum kernel matrix on real hardware (this is the slow step "
      "— can take minutes to hours depending on queue)...")
kernel_train = quantum_kernel.evaluate(x_vec=X_train)
kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

qsvm = SVC(kernel='precomputed', class_weight='balanced')
qsvm.fit(kernel_train, y_train)

y_pred = qsvm.predict(kernel_test)

# ---------------------------------------------------------
# 7. Evaluate — UNCHANGED
# ---------------------------------------------------------
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== QUANTUM KERNEL SVM RESULTS (IBM BACKEND) =====")
print(f"Backend   : {backend.name}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFull report:\n", classification_report(y_test, y_pred, digits=4, zero_division=0))

# ---------------------------------------------------------
# 8. Save results for Triveda's comparison notebook — UNCHANGED (+ backend name)
# ---------------------------------------------------------
results = {
    'model': 'Quantum Kernel SVM (ZZFeatureMap)',
    'backend': backend.name,
    'n_qubits': N_QUBITS,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'true_negatives': int(cm[0][0]),
    'false_positives': int(cm[0][1]),
    'false_negatives': int(cm[1][0]),
    'true_positives': int(cm[1][1]),
}

pd.DataFrame([results]).to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved metrics: {OUTPUT_PATH}")
print("Push this + the notebook to quantum/ in the repo.")