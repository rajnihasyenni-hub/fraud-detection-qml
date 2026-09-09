import numpy as np
from sklearn.metrics import average_precision_score


def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def compute_auprc(y_true, y_score):
    """Compute AUPRC for imbalanced fraud detection, where higher is better."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_false_negative_rate(y_true, y_pred):
    """Return the fraction of actual fraud cases that were missed."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.size == 0:
        return 0.0
    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.0
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    return float(false_negatives / positives)


def _energy_for_batch(model, X):
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    Xn = (X_arr - model["mean"]) / model["std"]
    hidden = _sigmoid(Xn @ model["W"] + model["hb"])
    reconstructed = _sigmoid(hidden @ model["W"].T + model["vb"])
    return np.mean(np.abs(Xn - reconstructed), axis=1)


def train_qrbm(X_train, y_train, n_hidden=8, learning_rate=0.05, epochs=40, seed=42):
    """Train a compact quantum-inspired restricted Boltzmann model.

    This implementation is intentionally lightweight and deterministic so that it can
    run reliably in a Streamlit web app and provide a real QRBM-style energy model
    without requiring a heavy external quantum stack.
    """
    X = np.asarray(X_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[0] == 0:
        raise ValueError("X_train must contain at least one sample.")

    rng = np.random.default_rng(seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    n_visible = Xn.shape[1]
    n_hidden = min(max(2, n_hidden), n_visible)

    W = rng.normal(0.0, 0.2, size=(n_visible, n_hidden))
    vb = np.zeros(n_visible)
    hb = np.zeros(n_hidden)

    for _ in range(int(epochs)):
        pos_hidden_prob = _sigmoid(Xn @ W + hb)
        pos_hidden_state = pos_hidden_prob > rng.random(pos_hidden_prob.shape)

        neg_visible_prob = _sigmoid(pos_hidden_state @ W.T + vb)
        neg_hidden_prob = _sigmoid(neg_visible_prob @ W + hb)

        pos_association = Xn.T @ pos_hidden_prob
        neg_association = neg_visible_prob.T @ neg_hidden_prob

        W += learning_rate * (pos_association - neg_association) / Xn.shape[0]
        vb += learning_rate * (Xn - neg_visible_prob).mean(axis=0)
        hb += learning_rate * (pos_hidden_prob - neg_hidden_prob).mean(axis=0)

    # Fix a threshold from the legitimate class if labels are supplied.
    legitimate_mask = np.asarray(y_train) == 0
    if legitimate_mask.any():
        valid_X = Xn[legitimate_mask]
        valid_hidden = _sigmoid(valid_X @ W + hb)
        valid_recon = _sigmoid(valid_hidden @ W.T + vb)
        energy = np.mean(np.abs(valid_X - valid_recon), axis=1)
        threshold = float(np.median(energy))
    else:
        threshold = float(np.median(np.abs(Xn - _sigmoid(_sigmoid(Xn @ W + hb) @ W.T + vb))))

    model = {
        "W": W,
        "vb": vb,
        "hb": hb,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "n_hidden": n_hidden,
    }
    return model


def predict_qrbm(model, X, return_probabilities=False):
    """Predict labels for a batch of feature vectors using the trained QRBM energy score.

    By default this returns only the fraud/legitimate labels. Set
    return_probabilities=True to also receive the soft confidence score.
    """
    energy = _energy_for_batch(model, X)
    pred = (energy > model["threshold"]).astype(int)
    probs = np.clip(1.0 / (1.0 + np.exp(-(energy - model["threshold"]))), 1e-6, 1 - 1e-6)

    if return_probabilities:
        return pred, probs, energy
    return pred
