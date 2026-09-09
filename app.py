"""
QuantumGuard — Demo Dashboard
Owner: whoever's presenting to judges

A lightweight Streamlit app to visually present the classical vs quantum
comparison during the hackathon demo. Reads the results CSVs your
teammates' scripts already produce — does NOT retrain anything live
(quantum kernel computation is too slow to run in front of judges).

Install:
    pip install streamlit pandas matplotlib --break-system-packages

Run:
    streamlit run app.py

Run this from the REPO ROOT (not inside a subfolder) so the relative
paths below find the results CSVs correctly.
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="QuantumGuard", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------
# Global styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1120 0%, #171a34 100%);
    }
    .qg-hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
        box-shadow: 0 10px 30px rgba(37, 117, 252, 0.35);
        margin-bottom: 1.5rem;
    }
    .qg-hero h1 {
        color: #ffffff;
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
    }
    .qg-hero p {
        color: #e6e6ff;
        font-size: 1.05rem;
        margin: 0;
    }
    .qg-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
        color: #ffffff;
        font-size: 0.8rem;
        margin-top: 0.6rem;
        margin-right: 0.4rem;
    }
    .qg-card {
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .qg-card h3 {
        margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Hero header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="qg-hero">
        <h1>🛡️ QuantumGuard</h1>
        <p>Quantum-Enhanced Credit Card Fraud Detection</p>
        <span class="qg-badge">Qiskit Fall Fest 2026</span>
        <span class="qg-badge">Centurion University of Technology and Management, Vizianagaram</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Problem statement
# ---------------------------------------------------------
with st.expander("📌 Problem Statement", expanded=True):
    st.write(
        "Fraud is rare relative to legitimate transactions, creating severe "
        "data imbalance that classical models handle imperfectly — causing "
        "false positives and missed fraud."
    )
    st.write(
        "**Our approach:** benchmark a classical Random Forest baseline "
        "against a quantum kernel SVM (built with a `ZZFeatureMap` and "
        "fidelity quantum kernel in Qiskit) to see whether quantum-encoded "
        "similarity can better separate fraud from legitimate transactions."
    )

st.markdown("---")

# ---------------------------------------------------------
# Helper to load a results CSV safely
# ---------------------------------------------------------
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

classical = load_csv("classical_baseline/classical_baseline_results.csv")
quantum = load_csv("quantum/quantum_kernel_results.csv")
comparison = load_csv("evaluation/final_comparison.csv")

# ---------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------
st.header("📊 Classical vs Quantum — Results")

if classical is None or quantum is None:
    st.warning(
        "Results not found yet. Run `baseline_model.py` and "
        "`quantum_kernel_svm.py` first, then `compare_results.py`, "
        "so their output CSVs exist for this dashboard to read."
    )
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="qg-card">', unsafe_allow_html=True)
        st.subheader("🌲 Classical (Random Forest)")
        st.metric("Precision", f"{classical['precision'].values[0]:.3f}")
        st.metric("Recall", f"{classical['recall'].values[0]:.3f}")
        st.metric("F1 Score", f"{classical['f1_score'].values[0]:.3f}")
        st.caption(f"Trained on full dataset ({classical.get('train_size', ['N/A']).values[0] if 'train_size' in classical else 'full'} rows)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="qg-card">', unsafe_allow_html=True)
        st.subheader("⚛️ Quantum Kernel SVM")
        st.metric("Precision", f"{quantum['precision'].values[0]:.3f}")
        st.metric("Recall", f"{quantum['recall'].values[0]:.3f}")
        st.metric("F1 Score", f"{quantum['f1_score'].values[0]:.3f}")
        st.caption(f"Trained on balanced subset ({quantum['train_size'].values[0]} samples), {quantum['n_qubits'].values[0]} qubits")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Side-by-side comparison")
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)
    else:
        chart_data = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1 Score"],
            "Classical (RF)": [
                classical['precision'].values[0],
                classical['recall'].values[0],
                classical['f1_score'].values[0],
            ],
            "Quantum Kernel SVM": [
                quantum['precision'].values[0],
                quantum['recall'].values[0],
                quantum['f1_score'].values[0],
            ],
        }).set_index("Metric")
        st.bar_chart(chart_data)

    st.info(
        "⚠️ Note: the classical model trains on the full ~284K-row dataset, "
        "while the quantum model trains on a small balanced subset due to "
        "current quantum simulator constraints. This is a fair "
        "proof-of-concept comparison, not a like-for-like benchmark."
    )

st.markdown("---")

# ---------------------------------------------------------
# Confusion matrices (if the PNGs exist)
# ---------------------------------------------------------
st.header("🔍 Confusion Matrices")

col1, col2 = st.columns(2)
classical_img = "classical_baseline/classical_confusion_matrix.png"
comparison_img = "evaluation/comparison_chart.png"

with col1:
    if os.path.exists(classical_img):
        st.image(classical_img, caption="Classical Baseline")
    else:
        st.caption("Confusion matrix image not found — run baseline_model.py first.")

with col2:
    if os.path.exists(comparison_img):
        st.image(comparison_img, caption="Classical vs Quantum Comparison")
    else:
        st.caption("Comparison chart not found — run compare_results.py first.")
