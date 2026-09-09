"""
Classical vs Quantum Comparison
Owner: Triveda

Reads the results CSVs produced by:
  - classical_baseline/baseline_model.py  -> classical_baseline_results.csv
  - quantum/quantum_kernel_svm.py         -> quantum_kernel_results.csv

Produces a side-by-side comparison table and bar chart for the final
presentation/README.

Run this AFTER both baseline_model.py and quantum_kernel_svm.py have
been run at least once (their CSVs need to exist).
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. Load both results files
# ---------------------------------------------------------
classical_path = "../classical_baseline/classical_baseline_results.csv"
quantum_path = "../quantum/quantum_kernel_results.csv"

if not os.path.exists(classical_path):
    raise FileNotFoundError(
        f"Can't find {classical_path}. Run classical_baseline/baseline_model.py first."
    )
if not os.path.exists(quantum_path):
    raise FileNotFoundError(
        f"Can't find {quantum_path}. Run quantum/quantum_kernel_svm.py first."
    )

classical = pd.read_csv(classical_path)
quantum = pd.read_csv(quantum_path)

# ---------------------------------------------------------
# 2. Build comparison table
# ---------------------------------------------------------
comparison = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Classical (Random Forest)': [
        classical['precision'].values[0],
        classical['recall'].values[0],
        classical['f1_score'].values[0],
    ],
    'Quantum Kernel SVM': [
        quantum['precision'].values[0],
        quantum['recall'].values[0],
        quantum['f1_score'].values[0],
    ],
})

print("===== CLASSICAL vs QUANTUM COMPARISON =====\n")
print(comparison.to_string(index=False))

comparison.to_csv("final_comparison.csv", index=False)
print("\nSaved: final_comparison.csv")

# ---------------------------------------------------------
# 3. Bar chart for the presentation/README
# ---------------------------------------------------------
metrics = comparison['Metric']
classical_scores = comparison['Classical (Random Forest)']
quantum_scores = comparison['Quantum Kernel SVM']

x = range(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([i - width/2 for i in x], classical_scores, width, label='Classical (RF)', color='#4C72B0')
ax.bar([i + width/2 for i in x], quantum_scores, width, label='Quantum Kernel SVM', color='#C44E52')

ax.set_ylabel('Score')
ax.set_title('Classical vs Quantum: Fraud Detection Performance')
ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=150)
print("Saved: comparison_chart.png")

# ---------------------------------------------------------
# 4. Important caveat to include in the writeup
# ---------------------------------------------------------
print("\nNOTE FOR THE README/PRESENTATION:")
print("The classical model was trained on the FULL dataset (284k+ rows).")
print("The quantum model was trained on a small balanced subset")
print(f"({quantum['train_size'].values[0]} train samples) due to simulator")
print("constraints. State this explicitly - it is not a like-for-like")
print("comparison, but it is a fair proof-of-concept comparison given")
print("current quantum hardware/simulator limitations.")
