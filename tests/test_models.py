"""
Sanity Tests — Fraud Detection Models
Owner: whoever's doing QA / final integration (Hasini or Triveda)

These are NOT rigorous unit tests — they're quick sanity checks to catch
broken imports, shape mismatches, or crashes before the hackathon demo.
Run this after the classical and quantum scripts have been set up.

Run with:
    python tests/test_models.py
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print("Running sanity checks...\n")

# ---------------------------------------------------------
# Test 1: Required libraries import correctly
# ---------------------------------------------------------
def test_imports():
    try:
        import sklearn
        import qiskit
        import qiskit_machine_learning
        print("[PASS] All required libraries import correctly")
        print(f"       qiskit version: {qiskit.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing library: {e}")
        print("       Run: pip install qiskit qiskit-machine-learning scikit-learn pandas --break-system-packages")
        return False


# ---------------------------------------------------------
# Test 2: Classical model pipeline works on synthetic data
# ---------------------------------------------------------
def test_classical_pipeline():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score

        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = np.zeros(200)
        y[:10] = 1  # 5% fraud, synthetic

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert len(preds) == len(y_test), "Prediction length mismatch"
        print("[PASS] Classical (Random Forest) pipeline runs end-to-end on synthetic data")
        return True
    except Exception as e:
        print(f"[FAIL] Classical pipeline broke: {e}")
        return False


# ---------------------------------------------------------
# Test 3: Quantum feature map + kernel builds without error
# ---------------------------------------------------------
def test_quantum_pipeline():
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        n_qubits = 4
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement='linear')

        assert feature_map.num_qubits == n_qubits, "Feature map qubit count mismatch"

        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

        # tiny synthetic batch - just checking it computes without crashing
        np.random.seed(42)
        X_small = np.random.uniform(0, np.pi, size=(5, n_qubits))
        kernel_matrix = quantum_kernel.evaluate(x_vec=X_small)

        assert kernel_matrix.shape == (5, 5), f"Unexpected kernel shape: {kernel_matrix.shape}"
        print("[PASS] Quantum feature map + kernel builds and computes correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Quantum pipeline broke: {e}")
        return False


# ---------------------------------------------------------
# Test 4: QRBM adapter loads and predicts without crashing
# ---------------------------------------------------------
def test_qrbm_pipeline():
    try:
        from quantum.qrbm import train_qrbm, predict_qrbm

        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 6))
        y = np.array([0] * 60 + [1] * 60)
        X = X + np.where(y[:, None] == 1, 2.0, -2.0)

        model = train_qrbm(X, y)
        preds = predict_qrbm(model, X[:10])

        assert len(preds) == 10, "Prediction length mismatch"
        print("[PASS] QRBM module loads and predicts correctly")
        return True
    except Exception as e:
        print(f"[FAIL] QRBM pipeline broke: {e}")
        return False


# ---------------------------------------------------------
# Test 5: QRBM energy metrics and threshold calibration work
# ---------------------------------------------------------
def test_qrbm_energy_metrics():
    try:
        from quantum.qrbm import compute_auprc, compute_false_negative_rate, train_qrbm

        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.05, 0.12, 0.88, 0.91, 0.15, 0.77])

        auprc = compute_auprc(y_true, scores)
        fnr = compute_false_negative_rate(y_true, np.array([0, 0, 1, 1, 0, 0]))

        assert 0.0 <= auprc <= 1.0, "AUPRC should stay between 0 and 1"
        assert 0.0 <= fnr <= 1.0, "FNR should stay between 0 and 1"

        rng = np.random.default_rng(7)
        X = rng.normal(size=(160, 6))
        y = np.array([0] * 120 + [1] * 40)
        X = X + np.where(y[:, None] == 1, 3.0, -1.0)

        model = train_qrbm(X, y)
        assert "threshold" in model and model["threshold"] >= 0.0, "QRBM threshold missing or invalid"

        print("[PASS] QRBM energy metrics and threshold calibration work")
        return True
    except Exception as e:
        print(f"[FAIL] QRBM energy metrics failed: {e}")
        return False


# ---------------------------------------------------------
# Test 6: Data file check (won't fail the suite, just warns)
# ---------------------------------------------------------
def test_dataset_present():
    import os
    possible_paths = ["creditcard.csv", "data/creditcard.csv", "../data/creditcard.csv"]
    found = any(os.path.exists(p) for p in possible_paths)
    if found:
        print("[PASS] creditcard.csv found")
    else:
        print("[WARN] creditcard.csv not found in expected locations.")
        print("       This is expected if you haven't downloaded it yet — see data/README.md")
    return True  # not a hard failure


# ---------------------------------------------------------
# Run all tests
# ---------------------------------------------------------
if __name__ == "__main__":
    results = []
    results.append(test_imports())
    results.append(test_classical_pipeline())
    results.append(test_quantum_pipeline())
    results.append(test_qrbm_pipeline())
    results.append(test_qrbm_energy_metrics())
    results.append(test_dataset_present())

    print("\n" + "=" * 50)
    if all(results):
        print("ALL SANITY CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - see [FAIL] lines above")
        sys.exit(1)
