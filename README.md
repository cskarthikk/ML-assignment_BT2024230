# Polynomial Regression Optimization (BT2024230)

This repository contains the complete machine learning pipeline for Assignment 1: Polynomial Regression. The goal is to accurately predict the target variable `y` for two datasets (`var1` and `var2`) using Polynomial Regression with mathematical regularization.

## Repository Structure

```text
BT2024230/
├── src/
│   ├── run_cv.py             # Performs 10-fold CV to find optimal degree & alpha using 1-SE rule
│   └── train_predict.py      # Fits final model on 100% of training data and generates predictions
├── submissions/
│   ├── BT2024230_pred_var1.csv   # Final 1000-row predictions for var1
│   └── BT2024230_pred_var2.csv   # Final 1000-row predictions for var2
├── report/
│   ├── report.pdf            # Detailed mathematical methodology and findings
│   └── plots/                # Validation curves supporting the chosen hyperparameters
└── README.md
```

## Methodology Summary
Due to the strict limits imposed by the problem (up to degree 10 for `var1` and degree 20 for `var2`), the feature matrix expands massively relative to the sample size ($p > n$ problem). Therefore, **Ridge Regression** is utilized to constrain the weights and prevent failing to generalize. 

A rigorous **10-fold Cross-Validation** strategy was used to search a wide, logarithmic grid of regularization penalties ($\alpha$), testing various preprocessing variants inside the cross-validation loops to prevent data leakage. The **One Standard Error (1-SE) rule** was subsequently applied to select the simplest parsimonious model that remained within one standard deviation of the absolute minimum validation error.

## Instructions to Run

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Run Cross Validation Search:**
   This script searches the optimal hyperparameters and saves the config files.
   ```bash
   cd src
   python run_cv.py
   ```

3. **Generate Final Predictions:**
   This script loads the best configs, trains the final models, and outputs the `.csv` files to the `submissions/` folder.
   ```bash
   cd src
   python train_predict.py
   ```
