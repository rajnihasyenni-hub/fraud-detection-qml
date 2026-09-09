"""
QuantumGuard — QRBM + Quantum Kernel SVM

ASSUMPTIONS:
1. Random Forest is removed.
2. Models shown by the UI are QRBM and Quantum Kernel SVM.
3. Existing QRBM code is assumed at quantum/qrbm.py and exposes:
       train_qrbm(X_train, y_train)
       predict_qrbm(model, X)
4. The existing QKSVM pipeline is retained.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

st.set_page_config(page_title="QuantumGuard — Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ QuantumGuard")
st.caption("Quantum-Enhanced Credit Card Fraud Detection · QRBM + Quantum Kernel SVM")

DATA_PATHS = ["creditcard.csv", "data/creditcard.csv", "../data/creditcard.csv"]

@st.cache_data
def load_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

df = load_data()
if df is None:
    st.error("creditcard.csv not found. Put it beside app.py or inside data/.")
    st.stop()

if "Class" not in df.columns:
    st.error("Dataset must contain a 'Class' column.")
    st.stop()

# ---------- shared preprocessing ----------
@st.cache_data
def prepare_features(data):
    d = data.copy()
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()
    d["Amount_scaled"] = amount_scaler.fit_transform(d[["Amount"]])
    d["Time_scaled"] = time_scaler.fit_transform(d[["Time"]])
    d = d.drop(["Amount", "Time"], axis=1)
    return d.drop("Class", axis=1), d["Class"].astype(int), amount_scaler, time_scaler

X_all, y_all, amount_scaler, time_scaler = prepare_features(df)

# ============================================================
# QRBM ADAPTER
# ============================================================
# This is intentionally an adapter, not a fabricated QRBM.
try:
    from quantum.qrbm import train_qrbm, predict_qrbm
    QRBM_IMPORT_ERROR = None
except Exception as exc:
    train_qrbm = None
    predict_qrbm = None
    QRBM_IMPORT_ERROR = exc

@st.cache_resource
def train_qrbm_model(X, y):
    if train_qrbm is None:
        raise ImportError(
            "Expected quantum/qrbm.py with train_qrbm(X_train, y_train)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = train_qrbm(X_train, y_train)
    result = predict_qrbm(model, X_test)
    predictions = result[0] if isinstance(result, tuple) else result
    predictions = np.asarray(predictions).reshape(-1).astype(int)

    metrics = {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    return model, metrics, X_train.columns.tolist()

def qrbm_predict_single(row, model, columns):
    x = row.copy()
    x["Amount_scaled"] = amount_scaler.transform(x[["Amount"]]).ravel()
    x["Time_scaled"] = time_scaler.transform(x[["Time"]]).ravel()
    x = x.drop(["Amount", "Time"], axis=1)[columns]
    result = predict_qrbm(model, x)
    if isinstance(result, tuple):
        pred = int(np.asarray(result[0]).reshape(-1)[0])
        try:
            prob = float(np.asarray(result[1]).reshape(-1)[0])
        except Exception:
            prob = None
    else:
        pred, prob = int(np.asarray(result).reshape(-1)[0]), None
    return pred, prob

# ============================================================
# QUANTUM KERNEL SVM
# ============================================================
N_QUBITS = 4
N_PER_CLASS = 80

@st.cache_resource
def train_quantum_model(data):
    fraud = data[data["Class"] == 1]
    legit = data[data["Class"] == 0]

    fraud = resample(fraud, n_samples=min(N_PER_CLASS, len(fraud)),
                     random_state=42, replace=False)
    legit = resample(legit, n_samples=min(N_PER_CLASS, len(legit)),
                     random_state=42, replace=False)

    balanced = pd.concat([fraud, legit]).sample(frac=1, random_state=42)
    X = balanced.drop("Class", axis=1)
    y = balanced["Class"].astype(int).values

    pca = PCA(n_components=N_QUBITS, random_state=42)
    reduced = pca.fit_transform(X)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaled = scaler.fit_transform(reduced)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled, y, test_size=.30, random_state=42, stratify=y
    )

    feature_map = ZZFeatureMap(
        feature_dimension=N_QUBITS, reps=2, entanglement="linear"
    )
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    K_train = kernel.evaluate(x_vec=X_train)
    K_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)

    qsvm = SVC(kernel="precomputed", class_weight="balanced", probability=True)
    qsvm.fit(K_train, y_train)
    pred = qsvm.predict(K_test)

    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

    return {
        "qsvm": qsvm, "kernel": kernel, "pca": pca, "scaler": scaler,
        "X_train": X_train, "feature_columns": X.columns.tolist()
    }, metrics

def quantum_predict_single(row, bundle):
    x = row[bundle["feature_columns"]]
    x = bundle["pca"].transform(x)
    x = bundle["scaler"].transform(x)
    K = bundle["kernel"].evaluate(x_vec=x, y_vec=bundle["X_train"])
    pred = int(bundle["qsvm"].predict(K)[0])
    prob = float(bundle["qsvm"].predict_proba(K)[0][1])
    return pred, prob

# ---------- training ----------
try:
    with st.spinner("Training QRBM..."):
        qrbm_model, qrbm_metrics, qrbm_columns = train_qrbm_model(X_all, y_all)
    qrbm_ready = True
except Exception as exc:
    qrbm_ready = False
    qrbm_model = None
    qrbm_columns = X_all.columns.tolist()
    qrbm_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    qrbm_error = exc

with st.spinner("Training Quantum Kernel SVM..."):
    quantum_bundle, quantum_metrics = train_quantum_model(df)

# ---------- sidebar ----------
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Live Detector"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Models**")
st.sidebar.write("⚛️ Quantum Restricted Boltzmann Machine")
st.sidebar.write("⚛️ Quantum Kernel SVM")
st.sidebar.markdown("---")
st.sidebar.markdown("**Team**")
st.sidebar.write("Monisha · Venu · Pratap · Triveda · Hasini")

# ============================================================
# DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.subheader("Model Performance Comparison")

    if not qrbm_ready:
        st.warning("QRBM could not be loaded. Random Forest is NOT used as a fallback.")
        st.code(str(qrbm_error))

    cols = st.columns(3)
    for col, label, key in zip(cols, ["PRECISION", "RECALL", "F1 SCORE"],
                               ["precision", "recall", "f1"]):
        col.metric(
            label,
            f"{qrbm_metrics[key]:.1%}  vs  {quantum_metrics[key]:.1%}"
        )

    fig = go.Figure()
    names = ["Precision", "Recall", "F1 Score"]
    fig.add_trace(go.Bar(
        name="Quantum RBM",
        x=names,
        y=[qrbm_metrics["precision"], qrbm_metrics["recall"], qrbm_metrics["f1"]]
    ))
    fig.add_trace(go.Bar(
        name="Quantum Kernel SVM",
        x=names,
        y=[quantum_metrics["precision"], quantum_metrics["recall"], quantum_metrics["f1"]]
    ))
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]), height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Comparison: QRBM vs Quantum Kernel SVM. No Random Forest is used.")

    fraud_count = int(df["Class"].sum())
    total = len(df)
    a, b, c = st.columns(3)
    a.metric("Total Transactions", f"{total:,}")
    b.metric("Fraud Cases", f"{fraud_count:,}")
    c.metric("Fraud Rate", f"{fraud_count / total:.3%}")

# ============================================================
# LIVE DETECTOR
# ============================================================
else:
    st.subheader("Try It Yourself")
    choice = st.radio(
        "Pick a sample:",
        ["Random legit transaction", "Random fraud transaction"],
        horizontal=True
    )

    if st.button("🎲 Load Transaction & Predict", type="primary"):
        fraud_pool = df[df["Class"] == 1].index.tolist()
        legit_pool = df[df["Class"] == 0].index.tolist()
        idx = np.random.choice(
            fraud_pool if choice == "Random fraud transaction" else legit_pool
        )

        row_full = df.loc[[idx]]
        row_features = row_full.drop("Class", axis=1)
        truth = int(row_full["Class"].iloc[0])

        if qrbm_ready:
            with st.spinner("Running QRBM prediction..."):
                qrbm_pred, qrbm_prob = qrbm_predict_single(
                    row_features, qrbm_model, qrbm_columns
                )
        else:
            qrbm_pred, qrbm_prob = None, None

        with st.spinner("Running quantum kernel evaluation..."):
            q_pred, q_prob = quantum_predict_single(row_features, quantum_bundle)

        a, b, c = st.columns(3)
        a.metric("Ground Truth", "FRAUD" if truth else "LEGITIMATE")
        b.metric("Quantum RBM",
                 "FRAUD" if qrbm_pred == 1 else "LEGITIMATE"
                 if qrbm_pred is not None else "Unavailable")
        c.metric("Quantum Kernel SVM",
                 "FRAUD" if q_pred == 1 else "LEGITIMATE")

        st.caption(f"Transaction amount: ₹{row_full['Amount'].iloc[0]:.2f}")

        if qrbm_prob is not None:
            st.write(f"QRBM confidence: {qrbm_prob:.1%}")
        st.write(f"Quantum Kernel SVM confidence: {q_prob:.1%}")

        if qrbm_pred is not None:
            if qrbm_pred == q_pred:
                st.success("Both quantum models agree.")
            else:
                st.warning("The two quantum models disagree.")
