"""
QuantumGuard — Quantum-Enhanced Credit Card Fraud Detection

This app combines a classical baseline, a quantum-inspired QRBM, and a quantum-kernel SVM.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

from quantum.qrbm import compute_auprc, compute_false_negative_rate, predict_qrbm, train_qrbm

# ============================================================
# PAGE CONFIG + STYLING
# ============================================================
st.set_page_config(
    page_title="QuantumGuard — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .hero { text-align: center; padding: 1.2rem 0 1rem 0; }
        .hero h1 {
            font-size: 3rem;
            background: linear-gradient(90deg, #7B61FF, #FF61D8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .hero p { color: #9CA3AF; font-size: 1.03rem; }
        .metric-card {
            background: rgba(123, 97, 255, 0.09);
            border: 1px solid rgba(123, 97, 255, 0.25);
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
        }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge-fraud {
            background: rgba(255, 72, 72, 0.15);
            color: #FF6B6B;
            border: 1px solid rgba(255, 72, 72, 0.4);
        }
        .badge-legit {
            background: rgba(72, 255, 150, 0.12);
            color: #4ADE80;
            border: 1px solid rgba(72, 255, 150, 0.4);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ QuantumGuard</h1>
        <p>Quantum-Enhanced Credit Card Fraud Detection · Qiskit Fall Fest 2026</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================
DATA_PATHS = ["creditcard.csv", "data/creditcard.csv", "../data/creditcard.csv"]


def find_data_path():
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    return None


@st.cache_data
def load_data():
    path = find_data_path()
    if path is None:
        return None
    return pd.read_csv(path)


df = load_data()
if df is None:
    st.error("creditcard.csv was not found. Put it next to app.py, inside data/, or in ../data/.")
    st.stop()

# ============================================================
# CLASSICAL MODEL
# ============================================================
@st.cache_resource
def train_classical_model(df):
    d = df.copy()
    scaler = StandardScaler()
    d["Amount_scaled"] = scaler.fit_transform(d[["Amount"]])
    d["Time_scaled"] = scaler.fit_transform(d[["Time"]])
    d = d.drop(["Amount", "Time"], axis=1)

    X = d.drop("Class", axis=1)
    y = d["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return model, metrics, (X_test, y_test)

# ============================================================
# QRBM MODEL
# ============================================================
N_QUBITS = 4


@st.cache_resource
def train_qrbm_model(df, n_qubits=N_QUBITS):
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    n_per_class = min(120, min(len(fraud), len(legit)))
    fraud_sample = resample(fraud, n_samples=n_per_class, random_state=42, replace=False)
    legit_sample = resample(legit, n_samples=n_per_class, random_state=42, replace=False)
    balanced = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)

    X = balanced.drop("Class", axis=1)
    y = balanced["Class"].astype(int).values

    pca = PCA(n_components=min(n_qubits, X.shape[1]), random_state=42)
    X_reduced = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.3, random_state=42, stratify=y
    )

    model = train_qrbm(X_train, y_train, n_hidden=6, learning_rate=0.08, epochs=50, seed=42)
    preds, _, energy_scores = predict_qrbm(model, X_test, return_probabilities=True)

    metrics = {
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "auprc": compute_auprc(y_test, energy_scores),
        "false_negative_rate": compute_false_negative_rate(y_test, preds),
        "threshold": float(model["threshold"]),
    }

    return {"model": model, "pca": pca, "feature_columns": X.columns.tolist()}, metrics


def qrbm_predict_single(row_features, bundle):
    row = row_features[bundle["feature_columns"]].copy()
    reduced = bundle["pca"].transform(row.to_frame().T)
    preds, probs, energy_scores = predict_qrbm(bundle["model"], reduced, return_probabilities=True)
    return int(preds[0]), float(probs[0]), float(energy_scores[0])

# ============================================================
# QUANTUM KERNEL SVM
# ============================================================
@st.cache_resource
def train_quantum_model(df, n_per_class=80, n_qubits=4):
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    fraud_sample = resample(fraud, n_samples=min(n_per_class, len(fraud)), random_state=42, replace=False)
    legit_sample = resample(legit, n_samples=n_per_class, random_state=42, replace=False)
    data = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)

    X = data.drop("Class", axis=1)
    y = data["Class"].values

    pca = PCA(n_components=n_qubits, random_state=42)
    X_reduced = pca.fit_transform(X)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_reduced)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    kernel_train = quantum_kernel.evaluate(x_vec=X_train)
    kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

    qsvm = SVC(kernel="precomputed", class_weight="balanced", probability=True)
    qsvm.fit(kernel_train, y_train)
    y_pred = qsvm.predict(kernel_test)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    bundle = {
        "qsvm": qsvm,
        "kernel": quantum_kernel,
        "pca": pca,
        "scaler": scaler,
        "X_train": X_train,
        "feature_columns": X.columns.tolist(),
    }
    return bundle, metrics


def quantum_predict_single(df_row_features, bundle):
    row = df_row_features[bundle["feature_columns"]]
    reduced = bundle["pca"].transform(row.to_frame().T)
    scaled = bundle["scaler"].transform(reduced)
    kernel_row = bundle["kernel"].evaluate(x_vec=scaled, y_vec=bundle["X_train"])
    pred = bundle["qsvm"].predict(kernel_row)[0]
    proba = bundle["qsvm"].predict_proba(kernel_row)[0][1]
    return int(pred), float(proba)

# ============================================================
# TRAIN MODELS
# ============================================================
with st.spinner("Training classical baseline..."):
    classical_model, classical_metrics, _ = train_classical_model(df)

with st.spinner("Training QRBM..."):
    qrbm_bundle, qrbm_metrics = train_qrbm_model(df)

with st.spinner("Training quantum kernel SVM..."):
    quantum_bundle, quantum_metrics = train_quantum_model(df)

# ============================================================
# SIDEBAR
# ============================================================
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Live Detector", "📘 Methodology"])
st.sidebar.markdown("---")
st.sidebar.markdown("**QuantumGuard Team**")
st.sidebar.write("Monisha · Venu · Pratap · Triveda · Hasini · Nihas")
st.sidebar.write("Qiskit Fall Fest 2026 · CUTM Vizianagaram")

# ============================================================
# DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.markdown("### Model Performance Comparison")

    metric_cols = st.columns(5)
    for col, label, value in zip(
        metric_cols,
        ["Precision", "Recall", "F1 Score", "AUPRC", "False Negative Rate"],
        [
            [classical_metrics["precision"], qrbm_metrics["precision"], quantum_metrics["precision"]],
            [classical_metrics["recall"], qrbm_metrics["recall"], quantum_metrics["recall"]],
            [classical_metrics["f1"], qrbm_metrics["f1"], quantum_metrics["f1"]],
            [0.0, qrbm_metrics["auprc"], 0.0],
            [0.0, qrbm_metrics["false_negative_rate"], 0.0],
        ],
    ):
        with col:
            if label in ["AUPRC", "False Negative Rate"]:
                display_value = value[1]
                formatted = f"{display_value:.2f}" if label == "AUPRC" else f"{display_value:.1%}"
                markup = f"""
                <div class="metric-card">
                    <div style="color:#9CA3AF;font-size:0.82rem;">{label}</div>
                    <div style="font-size:1.4rem;font-weight:700;">{formatted}</div>
                    <div style="color:#7B61FF;font-size:0.75rem;">QRBM only</div>
                </div>
                """
            else:
                markup = f"""
                <div class="metric-card">
                    <div style="color:#9CA3AF;font-size:0.82rem;">{label}</div>
                    <div style="font-size:1.5rem;font-weight:700;">
                        {value[0]:.1%} · {value[1]:.1%} · {value[2]:.1%}
                    </div>
                    <div style="color:#7B61FF;font-size:0.75rem;">Classical · QRBM · Quantum</div>
                </div>
                """
            st.markdown(markup, unsafe_allow_html=True)

    fig = go.Figure()
    names = ["Precision", "Recall", "F1 Score", "AUPRC", "FNR"]
    fig.add_trace(
        go.Bar(
            name="Classical",
            x=names,
            y=[classical_metrics["precision"], classical_metrics["recall"], classical_metrics["f1"], 0.0, 0.0],
            marker_color="#7B61FF",
        )
    )
    fig.add_trace(
        go.Bar(
            name="QRBM",
            x=names,
            y=[qrbm_metrics["precision"], qrbm_metrics["recall"], qrbm_metrics["f1"], qrbm_metrics["auprc"], qrbm_metrics["false_negative_rate"]],
            marker_color="#36CFC9",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Quantum Kernel SVM",
            x=names,
            y=[quantum_metrics["precision"], quantum_metrics["recall"], quantum_metrics["f1"], 0.0, 0.0],
            marker_color="#FF61D8",
        )
    )
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "This dashboard follows the README: a hybrid quantum-classical fraud detector that optimizes for recall, "
        "calibrates a QRBM energy threshold, and evaluates precision, F1, AUPRC, and false-negative rate under class imbalance."
    )

    fraud_count = int(df["Class"].sum())
    total = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases", f"{fraud_count:,}")
    c3.metric("Fraud Rate", f"{fraud_count / total:.3%}")

# ============================================================
# LIVE DETECTOR
# ============================================================
elif page == "🔍 Live Detector":
    st.markdown("### Predict on a Real Transaction")
    st.markdown("Select a real transaction and compare the classical model, QRBM, and quantum kernel SVM outputs.")

    fraud_pool = df[df["Class"] == 1].index.tolist()
    legit_pool = df[df["Class"] == 0].index.tolist()

    sample_type = st.radio(
        "Choose a transaction type:",
        ["Random legitimate transaction", "Random fraudulent transaction"],
        horizontal=True,
    )

    if st.button("Load Transaction & Predict", type="primary"):
        idx = np.random.choice(fraud_pool if sample_type == "Random fraudulent transaction" else legit_pool)
        row_full = df.loc[[idx]]
        true_label = int(row_full["Class"].values[0])
        row_features = row_full.drop("Class", axis=1)

        d_row = row_features.copy()
        d_row["Amount_scaled"] = d_row["Amount"]
        d_row["Time_scaled"] = d_row["Time"]
        d_row = d_row.drop(["Amount", "Time"], axis=1)
        d_row = d_row[classical_model.feature_names_in_]
        classical_pred = int(classical_model.predict(d_row)[0])
        classical_proba = float(classical_model.predict_proba(d_row)[0][1])

        with st.spinner("Running QRBM + quantum kernel evaluation..."):
            qrbm_pred, qrbm_proba, qrbm_energy = qrbm_predict_single(row_features, qrbm_bundle)
            quantum_pred, quantum_proba = quantum_predict_single(row_features, quantum_bundle)

        st.markdown("---")
        col_truth, col_classical, col_qrbm, col_quantum = st.columns(4)

        with col_truth:
            st.markdown("**Ground Truth**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if true_label == 1 else '<span class="badge badge-legit">LEGITIMATE</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Amount: ₹{row_full['Amount'].values[0]:.2f}")

        with col_classical:
            st.markdown("**Classical**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if classical_pred == 1 else '<span class="badge badge-legit">LEGITIMATE</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Confidence: {classical_proba:.1%}")

        with col_qrbm:
            st.markdown("**QRBM**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if qrbm_pred == 1 else '<span class="badge badge-legit">LEGITIMATE</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Energy score: {qrbm_energy:.4f}")
            st.caption(f"Threshold: {qrbm_bundle['model']['threshold']:.4f}")
            st.caption(f"Confidence: {qrbm_proba:.1%}")

        with col_quantum:
            st.markdown("**Quantum Kernel**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if quantum_pred == 1 else '<span class="badge badge-legit">LEGITIMATE</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Confidence: {quantum_proba:.1%}")

# ============================================================
# METHODOLOGY PAGE
# ============================================================
else:
    st.markdown("### Project Methodology")
    st.write(
        "This project follows the README concept for a hybrid quantum-classical fraud detection system: "
        "scale the transaction features, reduce the dimensionality to a compact quantum-feasible representation, "
        "calibrate a QRBM energy threshold, and compare different decision models under class imbalance."
    )

    st.markdown("#### 1. Data preparation")
    st.write("The app uses the anonymized credit-card dataset, with normalization for Amount and Time and balanced sampling for the quantum models to match the README’s class-imbalance handling strategy.")

    st.markdown("#### 2. Quantum feature encoding")
    st.write("The quantum kernel model uses a ZZFeatureMap to embed transaction patterns into a Hilbert-space representation suitable for quantum similarity analysis.")

    st.markdown("#### 3. QRBM energy model")
    st.write("The QRBM is implemented as a lightweight energy-based anomaly detector: it learns the normal transaction energy landscape and flags outliers above a calibrated threshold as potential fraud.")

    st.markdown("#### 4. Evaluation")
    st.write("The app reports precision, recall, F1 score, AUPRC, and false-negative rate to reflect the README’s focus on malignly imbalanced fraud detection and cost of missed fraud cases.")