import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import os

def load_config(var_name):
    with open(f"{var_name}_best_config.txt", "r") as f:
        d, a, prep = f.read().strip().split(",")
    return int(d), float(a), prep

def build_pipeline(d, a, prep):
    steps = [('poly', PolynomialFeatures(degree=d, include_bias=False))]
    if prep == 'C':
        # Scaler before poly
        steps.insert(0, ('scaler', StandardScaler()))
    elif prep == 'B':
        # Scaler after poly
        steps.append(('scaler', StandardScaler()))
    
    steps.append(('model', Ridge(alpha=a, solver='auto')))
    return Pipeline(steps)

def process_var(var_name):
    print(f"Processing {var_name}...")
    train_path = f"../BT2024230_train_{var_name}.csv"
    test_path = f"../BT2024230_test_{var_name}.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    y_train = train['y'].values
    if var_name == 'var1':
        features = ['x1','x2','x3','x4','x5','x6']
    else:
        features = ['x1','x2','x3']
        
    X_train = train[features].values
    X_test = test[features].values
    
    d, a, prep = load_config(var_name)
    print(f"Loaded config: Degree={d}, Alpha={a:.4f}, Prep={prep}")
    
    pipe = build_pipeline(d, a, prep)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    out_df = pd.DataFrame({'y': y_pred})
    
    # Assertions for correctness
    assert len(out_df) == len(test), f"Expected {len(test)} rows, got {len(out_df)}"
    assert list(out_df.columns) == ['y'], f"Expected column 'y', got {list(out_df.columns)}"
    assert not out_df.isna().any().any(), "Predictions contain NaN values"
    
    sub_dir = "../submissions"
    os.makedirs(sub_dir, exist_ok=True)
    
    out_path = os.path.join(sub_dir, f"BT2024230_pred_{var_name}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} and passed all validation assertions.")

if __name__ == "__main__":
    process_var('var1')
    process_var('var2')
