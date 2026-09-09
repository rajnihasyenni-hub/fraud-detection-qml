import numpy as np

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
)


# ============================================================
# Utility
# ============================================================

def _sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_auprc(y_true, scores):
    """
    Calculate Area Under the Precision-Recall Curve.
    """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.0

    return float(
        average_precision_score(y_true, scores)
    )


def compute_false_negative_rate(y_true, y_pred):
    """
    False Negative Rate = FN / (TP + FN)
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    actual_fraud = np.sum(y_true == 1)

    if actual_fraud == 0:
        return 0.0

    false_negatives = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    return float(false_negatives / actual_fraud)


# ============================================================
# Energy Function
# ============================================================

def _energy_for_batch(X, weights, hidden_bias, visible_bias):
    """
    Reconstruction-based anomaly energy.

    Higher energy = more anomalous.
    """

    X = np.asarray(X, dtype=float)

    hidden_prob = _sigmoid(
        X @ weights + hidden_bias
    )

    reconstruction = np.tanh(
        hidden_prob @ weights.T + visible_bias
    )

    energy = np.mean(
        (X - reconstruction) ** 2,
        axis=1
    )

    return energy


# ============================================================
# Probability Conversion
# ============================================================

def _energy_to_probability(energy, threshold):
    """
    Convert anomaly energy into a fraud probability.
    """

    scale = np.std(energy)

    if scale < 1e-12:
        scale = 1.0

    probability = _sigmoid(
        (energy - threshold) / scale
    )

    return probability


# ============================================================
# QRBM TRAINING
# ============================================================

def train_qrbm(
    X,
    y=None,
    n_hidden=4,
    learning_rate=0.03,
    epochs=40,
    seed=42,
):
    """
    Train a lightweight QRBM-style energy model.

    Parameters
    ----------
    X : array-like
        Training features.

    y : array-like, optional
        Training labels used for threshold calibration.

    n_hidden : int
        Number of hidden units.

    learning_rate : float
        Learning rate.

    epochs : int
        Number of training epochs.

    seed : int
        Random seed.

    Returns
    -------
    model : dict
        Trained QRBM model.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2-dimensional feature matrix."
        )

    rng = np.random.default_rng(seed)

    n_samples, n_features = X.shape

    n_hidden = max(
        2,
        min(int(n_hidden), 16)
    )

    # --------------------------------------------------------
    # Initialize parameters
    # --------------------------------------------------------

    weights = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(n_features, n_hidden)
    )

    hidden_bias = np.zeros(n_hidden)

    visible_bias = np.zeros(n_features)

    # --------------------------------------------------------
    # Contrastive-divergence-style training
    # --------------------------------------------------------

    for _ in range(int(epochs)):

        # Positive phase
        hidden_prob = _sigmoid(
            X @ weights + hidden_bias
        )

        # Reconstruction
        reconstruction = np.tanh(
            hidden_prob @ weights.T + visible_bias
        )

        # Negative phase
        hidden_reconstruction = _sigmoid(
            reconstruction @ weights + hidden_bias
        )

        # Gradients
        positive_gradient = (
            X.T @ hidden_prob
        ) / n_samples

        negative_gradient = (
            reconstruction.T
            @ hidden_reconstruction
        ) / n_samples

        # Parameter updates
        weights += learning_rate * (
            positive_gradient
            - negative_gradient
        )

        visible_bias += learning_rate * np.mean(
            X - reconstruction,
            axis=0
        )

        hidden_bias += learning_rate * np.mean(
            hidden_prob
            - hidden_reconstruction,
            axis=0
        )

    # --------------------------------------------------------
    # Calculate training energies
    # --------------------------------------------------------

    train_energy = _energy_for_batch(
        X,
        weights,
        hidden_bias,
        visible_bias
    )

    # Default threshold
    threshold = float(
        np.percentile(train_energy, 95)
    )

    # --------------------------------------------------------
    # Supervised threshold calibration
    # --------------------------------------------------------

    if y is not None:

        y = np.asarray(y).astype(int)

        if len(y) == len(train_energy):

            candidates = np.percentile(
                train_energy,
                np.linspace(70, 99, 30)
            )

            best_score = -1.0
            best_threshold = threshold

            for candidate in candidates:

                predictions = (
                    train_energy >= candidate
                ).astype(int)

                f1 = f1_score(
                    y,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y,
                    predictions,
                    zero_division=0
                )

                # Give extra importance to recall
                score = (
                    0.7 * f1
                    + 0.3 * recall
                )

                if score > best_score:

                    best_score = score
                    best_threshold = candidate

            threshold = float(best_threshold)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = {
        "weights": weights,
        "hidden_bias": hidden_bias,
        "visible_bias": visible_bias,
        "threshold": threshold,
        "n_hidden": n_hidden,
        "n_features": n_features,
    }

    return model


# ============================================================
# QRBM PREDICTION
# ============================================================

def predict_qrbm(
    model,
    X,
    return_probabilities=True,
):
    """
    Predict fraud using the trained QRBM.

    Returns
    -------
    predictions
        0 = legitimate
        1 = fraud

    probabilities
        Fraud probability for each transaction

    energies
        QRBM anomaly energy
    """

    X = np.asarray(X, dtype=float)

    weights = model["weights"]
    hidden_bias = model["hidden_bias"]
    visible_bias = model["visible_bias"]
    threshold = model["threshold"]

    # Calculate anomaly energy
    energies = _energy_for_batch(
        X,
        weights,
        hidden_bias,
        visible_bias
    )

    # Convert energy to probability
    probabilities = _energy_to_probability(
        energies,
        threshold
    )

    # Classification
    predictions = (
        energies >= threshold
    ).astype(int)

    if return_probabilities:
        return (
            predictions,
            probabilities,
            energies
        )

    return predictions


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Testing QRBM...")

    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(200, 4)
    )

    y = np.zeros(200, dtype=int)

    y[-20:] = 1

    model = train_qrbm(
        X,
        y,
        n_hidden=4,
        learning_rate=0.03,
        epochs=20,
        seed=42,
    )

    predictions, probabilities, energies = predict_qrbm(
        model,
        X,
        return_probabilities=True
    )

    print("QRBM IMPORT/TRAINING TEST PASSED")
    print(
        "AUPRC:",
        compute_auprc(y, probabilities)
    )
    print(
        "FNR:",
        compute_false_negative_rate(
            y,
            predictions
        )
    )