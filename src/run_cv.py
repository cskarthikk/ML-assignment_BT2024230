import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import time

def run_refined_cv(var_name, max_degree):
    print(f"\n--- Running Refined Search for {var_name} ---")
    
    train_path = f"../BT2024230_train_{var_name}.csv"
    if not os.path.exists(train_path):
        train_path = f"BT2024230_train_{var_name}.csv"
        
    train = pd.read_csv(train_path)
    
    if var_name == 'var1':
        X = train[['x1','x2','x3','x4','x5','x6']].values
    else:
        X = train[['x1','x2','x3']].values
    y = train['y'].values

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Refined broader grid
    alphas = np.logspace(-4, 2, 40)

    results = []

    for d in range(1, max_degree + 1):
        t0 = time.time()
        poly = PolynomialFeatures(degree=d, include_bias=False)
        
        # We will collect per-fold scores
        fold_scores = {f'{prep}_{a}': {'mse': [], 'r2': []} for prep in ['A', 'B', 'C'] for a in alphas}
        fold_scores['A_OLS_0'] = {'mse': [], 'r2': []}
            
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            # --- Variant C: Scaler Before Poly ---
            scaler_C = StandardScaler()
            X_tr_C = scaler_C.fit_transform(X_tr)
            X_va_C = scaler_C.transform(X_va)
            
            X_tr_poly_C = poly.fit_transform(X_tr_C)
            X_va_poly_C = poly.transform(X_va_C)
            
            # --- Variant A and B: Poly First ---
            X_tr_poly = poly.fit_transform(X_tr)
            X_va_poly = poly.transform(X_va)
            
            # Variant B: Scaler After Poly
            scaler_B = StandardScaler()
            X_tr_poly_B = scaler_B.fit_transform(X_tr_poly)
            X_va_poly_B = scaler_B.transform(X_va_poly)
            
            # OLS (Variant A)
            if d <= 5:
                model = LinearRegression()
                model.fit(X_tr_poly, y_tr)
                y_pred = model.predict(X_va_poly)
                fold_scores['A_OLS_0']['mse'].append(mean_squared_error(y_va, y_pred))
                fold_scores['A_OLS_0']['r2'].append(r2_score(y_va, y_pred))
                
            # Ridge
            for a in alphas:
                # Variant A
                model_A = Ridge(alpha=a, solver='auto')
                model_A.fit(X_tr_poly, y_tr)
                y_pred_A = model_A.predict(X_va_poly)
                fold_scores[f'A_{a}']['mse'].append(mean_squared_error(y_va, y_pred_A))
                fold_scores[f'A_{a}']['r2'].append(r2_score(y_va, y_pred_A))
                
                # Variant B
                model_B = Ridge(alpha=a, solver='auto')
                model_B.fit(X_tr_poly_B, y_tr)
                y_pred_B = model_B.predict(X_va_poly_B)
                fold_scores[f'B_{a}']['mse'].append(mean_squared_error(y_va, y_pred_B))
                fold_scores[f'B_{a}']['r2'].append(r2_score(y_va, y_pred_B))
                
                # Variant C
                model_C = Ridge(alpha=a, solver='auto')
                model_C.fit(X_tr_poly_C, y_tr)
                y_pred_C = model_C.predict(X_va_poly_C)
                fold_scores[f'C_{a}']['mse'].append(mean_squared_error(y_va, y_pred_C))
                fold_scores[f'C_{a}']['r2'].append(r2_score(y_va, y_pred_C))
                
        # Aggregate results
        if d <= 5:
            mse_mean = np.mean(fold_scores['A_OLS_0']['mse'])
            mse_std = np.std(fold_scores['A_OLS_0']['mse'], ddof=1)
            r2_mean = np.mean(fold_scores['A_OLS_0']['r2'])
            results.append({'degree': d, 'alpha': 0, 'prep': 'A_OLS', 'cv_mse': mse_mean, 'cv_mse_std': mse_std, 'cv_r2': r2_mean})
            
        for a in alphas:
            for prep in ['A', 'B', 'C']:
                mse_mean = np.mean(fold_scores[f'{prep}_{a}']['mse'])
                mse_std = np.std(fold_scores[f'{prep}_{a}']['mse'], ddof=1)
                r2_mean = np.mean(fold_scores[f'{prep}_{a}']['r2'])
                results.append({'degree': d, 'alpha': a, 'prep': prep, 'cv_mse': mse_mean, 'cv_mse_std': mse_std, 'cv_r2': r2_mean})
                
        print(f"Degree {d} done in {time.time()-t0:.1f}s")

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{var_name}_cv_results_refined.csv", index=False)
    
    # Apply 1-SE rule
    # Exclude A_OLS because train_predict.py expects a Ridge model pipeline
    valid_cands = res_df[res_df['prep'] != 'A_OLS']
    
    # 1. Find absolute minimum MSE
    best_overall = valid_cands.loc[valid_cands['cv_mse'].idxmin()]
    min_mse = best_overall['cv_mse']
    min_mse_std = best_overall['cv_mse_std']
    
    # Threshold for 1-SE (SE = std / sqrt(N))
    threshold = min_mse + (min_mse_std / np.sqrt(10))
    
    # 2. Find simplest model (lowest degree) within this threshold
    candidates_in_band = valid_cands[valid_cands['cv_mse'] <= threshold]
    min_deg = candidates_in_band['degree'].min()
    
    # 3. Break tie: get the model with the LOWEST MSE at that minimal degree
    winner = candidates_in_band[candidates_in_band['degree'] == min_deg].nsmallest(1, 'cv_mse').iloc[0]
    
    print(f"\n--- Best overall (Absolute Min) for {var_name} ---")
    print(best_overall)
    
    print(f"\n--- 1-SE Rule Winner for {var_name} ---")
    print(winner)
    
    # Calculate Train MSE/R2 for the 1-SE winner
    d = int(winner['degree'])
    a = float(winner['alpha'])
    prep = winner['prep']
    
    poly = PolynomialFeatures(degree=d, include_bias=False)
    
    if prep == 'A':
        X_poly = poly.fit_transform(X)
        model = Ridge(alpha=a, solver='auto')
        model.fit(X_poly, y)
        y_train_pred = model.predict(X_poly)
    elif prep == 'B':
        X_poly = poly.fit_transform(X)
        scaler = StandardScaler()
        X_poly = scaler.fit_transform(X_poly)
        model = Ridge(alpha=a, solver='auto')
        model.fit(X_poly, y)
        y_train_pred = model.predict(X_poly)
    elif prep == 'C':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_poly = poly.fit_transform(X_scaled)
        model = Ridge(alpha=a, solver='auto')
        model.fit(X_poly, y)
        y_train_pred = model.predict(X_poly)
        
    train_mse = mean_squared_error(y, y_train_pred)
    train_r2 = r2_score(y, y_train_pred)
    
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"CV MSE: {winner['cv_mse']:.4f} (+/- {winner['cv_mse_std']:.4f})")
    print(f"CV R2: {winner['cv_r2']:.4f}")

    # Write winner config to file for train_predict.py to read
    with open(f"{var_name}_best_config.txt", "w") as f:
        f.write(f"{d},{a},{prep}")
        
    # Plotting validation curve
    plt.figure(figsize=(10, 6))
    for p in ['A', 'B', 'C']:
        subset = res_df[res_df['prep'] == p]
        # Get best MSE per degree for this prep
        best_per_deg = subset.groupby('degree')['cv_mse'].min()
        plt.plot(best_per_deg.index, best_per_deg.values, marker='o', label=f'Variant {p}')
        
    # Add OLS if available
    ols = res_df[res_df['prep'] == 'A_OLS']
    if not ols.empty:
        plt.plot(ols['degree'], ols['cv_mse'], marker='x', linestyle='--', color='black', label='OLS')
        
    plt.yscale('log') # Log scale is often better for MSE
    plt.xlabel('Polynomial Degree')
    plt.ylabel('10-Fold CV MSE (Log Scale)')
    plt.title(f'Validation Curve: {var_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs('../report/plots', exist_ok=True)
    plt.savefig(f"../report/plots/{var_name}_val_curve.png")
    plt.close()

if __name__ == "__main__":
    run_refined_cv('var1', 10)
    run_refined_cv('var2', 20)
