import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

def generate_predictions():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    os.makedirs(sub_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Phase 1: var1
    # Best Model: Degree 5, Ridge alpha = 2.0309176, Variant A (Raw)
    # -------------------------------------------------------------
    print("--- Training and Predicting var1 ---")
    train1_path = os.path.join(base_dir, 'BT2024230_train_var1.csv')
    test1_path = os.path.join(base_dir, 'BT2024230_test_var1.csv')
    
    train1 = pd.read_csv(train1_path)
    test1 = pd.read_csv(test1_path)
    
    features1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    X1_train = train1[features1].values
    y1_train = train1['y'].values
    X1_test = test1[features1].values
    
    poly1 = PolynomialFeatures(degree=5, include_bias=False)
    X1_train_poly = poly1.fit_transform(X1_train)
    X1_test_poly = poly1.transform(X1_test)
    
    model1 = Ridge(alpha=2.030917620904739, solver='auto')
    model1.fit(X1_train_poly, y1_train)
    
    y1_pred = model1.predict(X1_test_poly)
    df1_sub = pd.DataFrame({'y': y1_pred})
    out1_path = os.path.join(sub_dir, 'BT2024230_pred_var1.csv')
    df1_sub.to_csv(out1_path, index=False)
    print(f"Saved: {out1_path} ({len(df1_sub)} rows)")
    print(f"Stats: Mean={y1_pred.mean():.3f}, Std={y1_pred.std():.3f}, Min={y1_pred.min():.3f}, Max={y1_pred.max():.3f}")
    
    # -------------------------------------------------------------
    # Phase 2: var2
    # Best Model: Degree 10, Ridge alpha = 1.4251027, Variant B (Scale Poly)
    # -------------------------------------------------------------
    print("\n--- Training and Predicting var2 ---")
    train2_path = os.path.join(base_dir, 'BT2024230_train_var2.csv')
    test2_path = os.path.join(base_dir, 'BT2024230_test_var2.csv')
    
    train2 = pd.read_csv(train2_path)
    test2 = pd.read_csv(test2_path)
    
    features2 = ['x1', 'x2', 'x3']
    X2_train = train2[features2].values
    y2_train = train2['y'].values
    X2_test = test2[features2].values
    
    poly2 = PolynomialFeatures(degree=10, include_bias=False)
    X2_train_poly = poly2.fit_transform(X2_train)
    X2_test_poly = poly2.transform(X2_test)
    
    scaler2 = StandardScaler()
    X2_train_poly_scaled = scaler2.fit_transform(X2_train_poly)
    X2_test_poly_scaled = scaler2.transform(X2_test_poly)
    
    model2 = Ridge(alpha=1.4251026703029992, solver='auto')
    model2.fit(X2_train_poly_scaled, y2_train)
    
    y2_pred = model2.predict(X2_test_poly_scaled)
    df2_sub = pd.DataFrame({'y': y2_pred})
    out2_path = os.path.join(sub_dir, 'BT2024230_pred_var2.csv')
    df2_sub.to_csv(out2_path, index=False)
    print(f"Saved: {out2_path} ({len(df2_sub)} rows)")
    print(f"Stats: Mean={y2_pred.mean():.3f}, Std={y2_pred.std():.3f}, Min={y2_pred.min():.3f}, Max={y2_pred.max():.3f}")

if __name__ == '__main__':
    generate_predictions()
