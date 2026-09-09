# Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Why it's not in this repo
The file is ~143MB — too large to comfortably commit to GitHub. It's
excluded via `.gitignore`.

## How to get it
1. Go to the Kaggle link above (free Kaggle account required)
2. Download `creditcard.csv`
3. Place it in this `data/` folder (or the repo root — both scripts
   look for `creditcard.csv` relative to where you run them from)

## About the dataset
- 284,807 transactions made by European cardholders in September 2013
- 492 are fraudulent (~0.17%) — heavily imbalanced
- Features V1–V28 are PCA-transformed (original features anonymized
  for confidentiality); only `Time` and `Amount` are untransformed
- `Class` column: 1 = fraud, 0 = legitimate
