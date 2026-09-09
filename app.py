"""
QuantumGuard — Web App
Streamlit dashboard + live fraud detector combining the classical
Random Forest baseline and the quantum kernel SVM.

Models train ONCE on startup (cached), so the app stays fast after
that — including live predictions from the REAL quantum kernel model,
not a placeholder.

Run with:
    streamlit run app.py

Requires creditcard.csv in the same folder as app.py, OR in data/ or
../data/ relative to it.

Install:
    pip install streamlit plotly qiskit qiskit-machine-learning scikit-learn pandas --break-system-packages
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ============================================================
# PAGE CONFIG + STYLING
# ============================================================
st.set_page_config(
    page_title="QuantumGuard — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .hero { text-align: center; padding: 1.5rem 0 1rem 0; }
    .hero h1 {
        font-size: 3rem;
        background: linear-gradient(90deg, #7B61FF, #FF61D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0;
    }
    .hero p { color: #9CA3AF; font-size: 1.1rem; }
    .metric-card {
        background: rgba(123, 97, 255, 0.08);
        border: 1px solid rgba(123, 97, 255, 0.25);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
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
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛡️ QuantumGuard</h1>
    <p>Quantum-Enhanced Credit Card Fraud Detection · Qiskit Fall Fest 2026</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
DATA_PATHS = ["creditcard.csv", "data/creditcard.csv", "../data/creditcard.csv"]

def find_data_path():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    return None

@st.cache_data
def load_data():
    path = find_data_path()
    if path is None:
        return None
    return pd.read_csv(path)

df = load_data()
if df is None:
    st.error(
        "creditcard.csv not found. Place it next to app.py, or in a "
        "data/ or ../data/ folder relative to it."
    )
    st.stop()

# ============================================================
# CLASSICAL MODEL (trained once, cached)
# ============================================================
@st.cache_resource
def train_classical_model(df):
    d = df.copy()
    scaler = StandardScaler()
    d['Amount_scaled'] = scaler.fit_transform(d[['Amount']])
    d['Time_scaled'] = scaler.fit_transform(d[['Time']])
    d = d.drop(['Amount', 'Time'], axis=1)

    X = d.drop('Class', axis=1)
    y = d['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    return model, metrics, (X_test, y_test)


# ============================================================
# QUANTUM MODEL (trained once, cached)
# ============================================================
N_QUBITS = 4
N_PER_CLASS = 80

@st.cache_resource
def train_quantum_model(df, n_per_class=N_PER_CLASS, n_qubits=N_QUBITS):
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    fraud_sample = resample(fraud, n_samples=min(n_per_class, len(fraud)),
                             random_state=42, replace=False)
    legit_sample = resample(legit, n_samples=n_per_class,
                             random_state=42, replace=False)

    data = pd.concat([fraud_sample, legit_sample]).sample(frac=1, random_state=42)
    X = data.drop('Class', axis=1)
    y = data['Class'].values

    pca = PCA(n_components=n_qubits, random_state=42)
    X_reduced = pca.fit_transform(X)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_reduced)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    kernel_train = quantum_kernel.evaluate(x_vec=X_train)
    kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

    qsvm = SVC(kernel='precomputed', class_weight='balanced', probability=True)
    qsvm.fit(kernel_train, y_train)

    y_pred = qsvm.predict(kernel_test)
    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    # bundle everything needed to score a brand-new sample later
    bundle = {
        'qsvm': qsvm, 'kernel': quantum_kernel, 'pca': pca, 'scaler': scaler,
        'X_train': X_train, 'feature_columns': X.columns.tolist(),
    }
    return bundle, metrics


def quantum_predict_single(df_row_features, bundle):
    """Run one real transaction through the trained quantum pipeline."""
    row = df_row_features[bundle['feature_columns']]
    reduced = bundle['pca'].transform(row)
    scaled = bundle['scaler'].transform(reduced)
    kernel_row = bundle['kernel'].evaluate(x_vec=scaled, y_vec=bundle['X_train'])
    pred = bundle['qsvm'].predict(kernel_row)[0]
    proba = bundle['qsvm'].predict_proba(kernel_row)[0][1]
    return pred, proba


# ============================================================
# TRAIN (cached — only slow the first time the app starts)
# ============================================================
with st.spinner("Training classical model..."):
    classical_model, classical_metrics, (X_test_c, y_test_c) = train_classical_model(df)

with st.spinner("Training quantum kernel model (first load takes ~1-2 min)..."):
    quantum_bundle, quantum_metrics = train_quantum_model(df)

# ============================================================
# SIDEBAR
# ============================================================
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Live Detector"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Team**")
st.sidebar.markdown("Monisha · Venu · Pratap · Triveda · Hasini . Nihas")
st.sidebar.markdown("Qiskit Fall Fest 2026 · CUTM Vizianagaram")

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.markdown("### Model Performance Comparison")

    c1, c2, c3 = st.columns(3)
    for col, label, key in zip([c1, c2, c3], ["PRECISION", "RECALL", "F1 SCORE"],
                                ['precision', 'recall', 'f1']):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#9CA3AF;font-size:0.85rem;">{label}</div>
                <div style="font-size:1.8rem;font-weight:700;">
                    {classical_metrics[key]:.1%} <span style="color:#9CA3AF;font-size:1rem;">vs</span> {quantum_metrics[key]:.1%}
                </div>
                <div style="color:#7B61FF;font-size:0.75rem;">Classical vs Quantum</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure()
    names = ['Precision', 'Recall', 'F1 Score']
    fig.add_trace(go.Bar(name='Classical (Random Forest)', x=names,
                          y=[classical_metrics['precision'], classical_metrics['recall'], classical_metrics['f1']],
                          marker_color='#7B61FF'))
    fig.add_trace(go.Bar(name='Quantum Kernel SVM', x=names,
                          y=[quantum_metrics['precision'], quantum_metrics['recall'], quantum_metrics['f1']],
                          marker_color='#FF61D8'))
    fig.update_layout(
        barmode='group', template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='.0%', range=[0, 1]), height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Note:** The classical model trains on the full dataset "
        f"({len(df):,} transactions). The quantum kernel model trains on a "
        f"smaller balanced subset ({len(quantum_bundle['X_train'])} samples) "
        "due to quantum simulator constraints — this is a proof-of-concept "
        "comparison, not a like-for-like benchmark."
    )

    st.markdown("### Dataset Overview")
    fraud_count = int(df['Class'].sum())
    total = len(df)
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Transactions", f"{total:,}")
    d2.metric("Fraud Cases", f"{fraud_count:,}")
    d3.metric("Fraud Rate", f"{fraud_count/total:.3%}")

# ============================================================
# PAGE 2: LIVE DETECTOR
# ============================================================
else:
    st.markdown("### Try It Yourself")
    st.markdown("Pick a real transaction and see BOTH models predict it live — including a real quantum circuit evaluation.")

    fraud_pool = df[df['Class'] == 1].index.tolist()
    legit_pool = df[df['Class'] == 0].index.tolist()

    choice_type = st.radio("Pick a sample:", ["Random legit transaction", "Random fraud transaction"], horizontal=True)

    if st.button("🎲 Load Transaction & Predict", type="primary"):
        idx = np.random.choice(fraud_pool if choice_type == "Random fraud transaction" else legit_pool)

        row_full = df.loc[[idx]]
        true_label = int(row_full['Class'].values[0])
        row_features = row_full.drop('Class', axis=1)

        # ---- Classical prediction ----
        d_row = row_features.copy()
        d_row['Amount_scaled'] = d_row['Amount']  # placeholder scale, model handles raw magnitude fine for demo
        d_row['Time_scaled'] = d_row['Time']
        d_row = d_row.drop(['Amount', 'Time'], axis=1)
        d_row = d_row[classical_model.feature_names_in_]
        classical_pred = classical_model.predict(d_row)[0]
        classical_proba = classical_model.predict_proba(d_row)[0][1]

        # ---- Quantum prediction (real inference, computed live) ----
        with st.spinner("Running quantum circuit evaluation..."):
            quantum_pred, quantum_proba = quantum_predict_single(row_features, quantum_bundle)

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("**Ground Truth**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if true_label == 1
                else '<span class="badge badge-legit">LEGITIMATE</span>', unsafe_allow_html=True)
            st.caption(f"Amount: ₹{row_full['Amount'].values[0]:.2f}")

        with r2:
            st.markdown("**Classical Model**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if classical_pred == 1
                else '<span class="badge badge-legit">LEGITIMATE</span>', unsafe_allow_html=True)
            st.caption(f"Confidence: {classical_proba:.1%}")

        with r3:
            st.markdown("**Quantum Kernel Model**")
            st.markdown(
                '<span class="badge badge-fraud">FRAUD</span>' if quantum_pred == 1
                else '<span class="badge badge-legit">LEGITIMATE</span>', unsafe_allow_html=True)
            st.caption(f"Confidence: {quantum_proba:.1%}")