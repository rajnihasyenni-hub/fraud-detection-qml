"""
Classical Baseline — Credit Card Fraud Detection
Owner: Monisha
Purpose: Train a classical model (Random Forest) on the FULL imbalanced
dataset and report precision/recall/F1/confusion matrix. This is the
benchmark that the quantum kernel model (Pratap/Venu) gets compared against.

Dataset: Kaggle "Credit Card Fraud Detection"
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download creditcard.csv and place it in the same folder as this script.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print("Fraud cases:", df['Class'].sum(), "out of", len(df))
print("Fraud %:", round(df['Class'].mean() * 100, 4), "%")

# ---------------------------------------------------------
# 2. Basic preprocessing
# ---------------------------------------------------------
# 'Amount' and 'Time' are not PCA-transformed like V1-V28, so scale them
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])

df = df.drop(['Amount', 'Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# ---------------------------------------------------------
# 3. Train/test split (stratified — keep fraud ratio consistent)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape, "  Test shape:", X_test.shape)
print("Fraud in train:", y_train.sum(), "  Fraud in test:", y_test.sum())

# ---------------------------------------------------------
# 4. Train Random Forest
# ---------------------------------------------------------
# class_weight='balanced' tells the model to pay extra attention to the
# rare fraud class instead of ignoring it (this is the classical way of
# handling imbalance, vs. SMOTE/undersampling which the quantum team
# will use on their smaller subset).
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Predict & evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n===== CLASSICAL BASELINE RESULTS (Random Forest) =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nFull report:\n", classification_report(y_test, y_pred, digits=4))

# ---------------------------------------------------------
# 6. Confusion matrix plot (for the slide/report)
# ---------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Classical Baseline (Random Forest) — Confusion Matrix')
plt.tight_layout()
plt.savefig('classical_confusion_matrix.png', dpi=150)
print("\nSaved plot: classical_confusion_matrix.png")

# ---------------------------------------------------------
# 7. Save metrics to a file so Triveda can pull them for comparison
# ---------------------------------------------------------
results = {
    'model': 'Random Forest (classical baseline)',
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'true_negatives': int(cm[0][0]),
    'false_positives': int(cm[0][1]),
    'false_negatives': int(cm[1][0]),
    'true_positives': int(cm[1][1]),
}

results_df = pd.DataFrame([results])
results_df.to_csv('classical_baseline_results.csv', index=False)
print("\nSaved metrics: classical_baseline_results.csv")
print("\nDone. Push this file + the CSV + the PNG to the repo under classical_baseline/")
