"""
QuantumGuard
Quantum-Enhanced Credit Card Fraud Detection

Streamlit dashboard for:
- Credit card fraud detection
- PCA dimensionality reduction
- Quantum feature encoding
- QRBM-style energy anomaly detection
- AUPRC / F1 / Recall / FNR evaluation
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from quantum.qrbm import (
    compute_auprc,
    compute_false_negative_rate,
    predict_qrbm,
    train_qrbm,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="QuantumGuard",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ QuantumGuard")

st.subheader(
    "Quantum-Enhanced Credit Card Fraud Detection"
)

st.write(
    """
    A hybrid classical–quantum inspired pipeline for detecting
    rare fraudulent transactions using dimensionality reduction,
    quantum feature encoding and energy-based anomaly detection.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Configuration")

n_components = st.sidebar.slider(
    "PCA components",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
)

n_hidden = st.sidebar.slider(
    "QRBM hidden units",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
)

epochs = st.sidebar.slider(
    "Training epochs",
    min_value=10,
    max_value=100,
    value=40,
    step=10,
)

learning_rate = st.sidebar.slider(
    "Learning rate",
    min_value=0.005,
    max_value=0.10,
    value=0.03,
    step=0.005,
)

test_size = st.sidebar.slider(
    "Test size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)


# ============================================================
# DATA LOADING
# ============================================================

def generate_demo_data(
    n_samples=2000,
    n_features=12,
    fraud_rate=0.05,
    seed=42,
):
    """
    Generate a synthetic credit-card-like dataset.

    This allows the dashboard to run even before the real
    Kaggle Credit Card Fraud Detection dataset is added.
    """

    rng = np.random.default_rng(seed)

    n_fraud = max(
        1,
        int(n_samples * fraud_rate)
    )

    n_legitimate = (
        n_samples - n_fraud
    )

    legitimate = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_legitimate, n_features),
    )

    fraud = rng.normal(
        loc=1.8,
        scale=1.3,
        size=(n_fraud, n_features),
    )

    X = np.vstack(
        [
            legitimate,
            fraud,
        ]
    )

    y = np.concatenate(
        [
            np.zeros(n_legitimate),
            np.ones(n_fraud),
        ]
    )

    indices = rng.permutation(
        len(X)
    )

    X = X[indices]
    y = y[indices]

    columns = [
        f"V{i + 1}"
        for i in range(n_features)
    ]

    df = pd.DataFrame(
        X,
        columns=columns,
    )

    df["Class"] = y.astype(int)

    return df


def load_uploaded_csv(uploaded_file):

    return pd.read_csv(
        uploaded_file
    )


# ============================================================
# DATA SECTION
# ============================================================

st.header("1. Data Preparation")

uploaded_file = st.file_uploader(
    "Upload Credit Card Fraud CSV",
    type=["csv"],
)

if uploaded_file is not None:

    try:

        df = load_uploaded_csv(
            uploaded_file
        )

        st.success(
            "Dataset loaded successfully."
        )

    except Exception as exc:

        st.error(
            f"Could not read CSV: {exc}"
        )

        st.stop()

else:

    df = generate_demo_data()

    st.info(
        """
        No CSV uploaded. QuantumGuard is currently using
        a synthetic imbalanced fraud dataset for demonstration.

        Upload the Kaggle Credit Card Fraud Detection CSV
        to run the pipeline on real data.
        """
    )


# ============================================================
# FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "Class",
    "class",
    "Fraud",
    "fraud",
    "label",
    "Label",
]

target_column = None

for column in target_candidates:

    if column in df.columns:

        target_column = column
        break


if target_column is None:

    st.error(
        """
        Target column not found.

        Your dataset must contain a fraud label such as:
        Class, class, Fraud, fraud, label or Label.
        """
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan,
)

df = df.dropna(
    axis=0
)

# Remove duplicate rows.
df = df.drop_duplicates()


# ============================================================
# NUMERIC FEATURES
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column != target_column
    and pd.api.types.is_numeric_dtype(
        df[column]
    )
]

if len(feature_columns) < 2:

    st.error(
        "At least two numeric feature columns are required."
    )

    st.stop()


X = df[feature_columns].values

y = (
    df[target_column]
    .astype(int)
    .values
)


# ============================================================
# DATASET INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Transactions",
        f"{len(df):,}"
    )

with col2:

    fraud_count = int(
        np.sum(y == 1)
    )

    st.metric(
        "Fraud cases",
        f"{fraud_count:,}"
    )

with col3:

    legitimate_count = int(
        np.sum(y == 0)
    )

    st.metric(
        "Legitimate cases",
        f"{legitimate_count:,}"
    )

with col4:

    fraud_rate = (
        np.mean(y == 1) * 100
    )

    st.metric(
        "Fraud rate",
        f"{fraud_rate:.2f}%"
    )


with st.expander(
    "View dataset"
):

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.subheader(
    "Class Distribution"
)

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

fig_class = go.Figure()

fig_class.add_trace(
    go.Bar(
        x=[
            "Legitimate",
            "Fraud",
        ],
        y=[
            class_counts.get(0, 0),
            class_counts.get(1, 0),
        ],
    )
)

fig_class.update_layout(
    title="Transaction Class Distribution",
    xaxis_title="Class",
    yaxis_title="Number of Transactions",
)

st.plotly_chart(
    fig_class,
    use_container_width=True,
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

st.header(
    "2. Classical Pre-processing"
)

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# PCA
# ============================================================

actual_components = min(
    n_components,
    X_train_scaled.shape[1],
    X_train_scaled.shape[0],
)

pca = PCA(
    n_components=actual_components,
    random_state=42,
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

explained_variance = (
    np.sum(
        pca.explained_variance_ratio_
    )
    * 100
)


col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Original features",
        X_train.shape[1],
    )

with col2:

    st.metric(
        "PCA components",
        actual_components,
    )

st.info(
    f"PCA retains approximately "
    f"{explained_variance:.2f}% of the variance."
)


# ============================================================
# PCA VISUALIZATION
# ============================================================

if actual_components >= 2:

    fig_pca = go.Figure()

    normal_mask = (
        y_test == 0
    )

    fraud_mask = (
        y_test == 1
    )

    fig_pca.add_trace(
        go.Scatter(
            x=X_test_pca[normal_mask, 0],
            y=X_test_pca[normal_mask, 1],
            mode="markers",
            name="Legitimate",
            marker=dict(
                size=5
            ),
        )
    )

    fig_pca.add_trace(
        go.Scatter(
            x=X_test_pca[fraud_mask, 0],
            y=X_test_pca[fraud_mask, 1],
            mode="markers",
            name="Fraud",
            marker=dict(
                size=7
            ),
        )
    )

    fig_pca.update_layout(
        title="PCA Feature Space",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True,
    )


# ============================================================
# QUANTUM FEATURE ENCODING
# ============================================================

st.header(
    "3. Quantum Feature Encoding"
)

st.write(
    """
    PCA-compressed features are scaled into the interval
    [0, π] so they can be mapped to quantum rotation angles.
    """
)

angle_scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_angles = angle_scaler.fit_transform(
    X_train_pca
)

X_test_angles = angle_scaler.transform(
    X_test_pca
)


st.write(
    f"Quantum feature dimension: "
    f"**{X_train_angles.shape[1]} qubits/features**"
)


# ============================================================
# OPTIONAL QISKIT ENCODING
# ============================================================

def create_quantum_feature_circuit(features):

    try:

        from qiskit import QuantumCircuit

        n_qubits = len(features)

        circuit = QuantumCircuit(
            n_qubits
        )

        for i, angle in enumerate(features):

            circuit.ry(
                float(angle),
                i,
            )

        # Simple entanglement layer.
        for i in range(
            n_qubits - 1
        ):

            circuit.cx(
                i,
                i + 1,
            )

        return circuit

    except Exception:

        return None


if st.checkbox(
    "Show Qiskit quantum feature circuit"
):

    sample_angles = X_test_angles[0]

    circuit = create_quantum_feature_circuit(
        sample_angles
    )

    if circuit is not None:

        st.code(
            str(circuit.draw()),
            language="text",
        )

    else:

        st.warning(
            "Qiskit is not installed. "
            "The classical QRBM energy model can still run."
        )


# ============================================================
# QRBM TRAINING
# ============================================================

st.header(
    "4. QRBM Energy-Based Model"
)

st.write(
    """
    QuantumGuard learns the energy landscape of the
    transaction data. Legitimate transactions should have
    lower reconstruction energy, while anomalous transactions
    should have higher energy.
    """
)

train_button = st.button(
    "Train QuantumGuard",
    type="primary",
)


if train_button:

    with st.spinner(
        "Training QRBM energy model..."
    ):

        model = train_qrbm(
            X_train_angles,
            y_train,
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            epochs=epochs,
            seed=42,
        )

        predictions, probabilities, energies = predict_qrbm(
            model,
            X_test_angles,
            return_probabilities=True,
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    auprc = compute_auprc(
        y_test,
        probabilities,
    )

    fnr = compute_false_negative_rate(
        y_test,
        predictions,
    )

    threshold = model[
        "threshold"
    ]

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    st.session_state[
        "model"
    ] = model

    st.session_state[
        "predictions"
    ] = predictions

    st.session_state[
        "probabilities"
    ] = probabilities

    st.session_state[
        "energies"
    ] = energies

    st.session_state[
        "metrics"
    ] = {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUPRC": auprc,
        "False Negative Rate": fnr,
    }

    st.session_state[
        "threshold"
    ] = threshold


# ============================================================
# RESULTS
# ============================================================

if "metrics" in st.session_state:

    st.header(
        "5. Fraud Detection Results"
    )

    metrics = st.session_state[
        "metrics"
    ]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Precision",
            f"{metrics['Precision']:.4f}"
        )

    with col2:

        st.metric(
            "Recall",
            f"{metrics['Recall']:.4f}"
        )

    with col3:

        st.metric(
            "F1-score",
            f"{metrics['F1-score']:.4f}"
        )

    with col4:

        st.metric(
            "AUPRC",
            f"{metrics['AUPRC']:.4f}"
        )

    with col5:

        st.metric(
            "False Negative Rate",
            f"{metrics['False Negative Rate']:.4f}"
        )

    st.divider()

    # --------------------------------------------------------
    # Energy Distribution
    # --------------------------------------------------------

    energies = st.session_state[
        "energies"
    ]

    predictions = st.session_state[
        "predictions"
    ]

    threshold = st.session_state[
        "threshold"
    ]

    fig_energy = go.Figure()

    fig_energy.add_trace(
        go.Histogram(
            x=energies[y_test == 0],
            name="Legitimate",
            opacity=0.7,
        )
    )

    fig_energy.add_trace(
        go.Histogram(
            x=energies[y_test == 1],
            name="Fraud",
            opacity=0.7,
        )
    )

    fig_energy.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text="Fraud Threshold",
    )

    fig_energy.update_layout(
        title="QRBM Energy Distribution",
        xaxis_title="Energy Score",
        yaxis_title="Transactions",
        barmode="overlay",
    )

    st.plotly_chart(
        fig_energy,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                "Predicted Legitimate",
                "Predicted Fraud",
            ],
            y=[
                "Actual Legitimate",
                "Actual Fraud",
            ],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )

    fig_cm.update_layout(
        title="Fraud Detection Confusion Matrix"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Prediction Table
    # --------------------------------------------------------

    st.subheader(
        "Transaction Predictions"
    )

    results_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Prediction": predictions,
            "Energy Score": energies,
            "Fraud Probability": st.session_state[
                "probabilities"
            ],
        }
    )

    results_df[
        "Actual"
    ] = results_df[
        "Actual"
    ].map(
        {
            0: "Legitimate",
            1: "Fraud",
        }
    )

    results_df[
        "Prediction"
    ] = results_df[
        "Prediction"
    ].map(
        {
            0: "Legitimate",
            1: "Fraud",
        }
    )

    st.dataframe(
        results_df.head(100),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Download Results
    # --------------------------------------------------------

    csv = results_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Predictions CSV",
        data=csv,
        file_name="quantumguard_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header(
    "About QuantumGuard"
)

st.markdown(
    """
    **QuantumGuard** is a hackathon prototype for
    quantum-enhanced credit-card fraud detection.

    ### Pipeline

    `Raw Transactions`
    → `Scaling`
    → `PCA`
    → `Quantum Feature Encoding`
    → `QRBM Energy Model`
    → `Energy Score`
    → `Threshold`
    → `Fraud / Legitimate`

    ### Evaluation Metrics

    - Precision
    - Recall
    - F1-score
    - AUPRC
    - False Negative Rate

    The system emphasizes recall because missed fraudulent
    transactions can be more costly than false alarms.
    """
)