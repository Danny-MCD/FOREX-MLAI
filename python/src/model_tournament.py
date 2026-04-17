import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# --- SETTINGS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data_vault")
MODEL_DIR = os.path.join(BASE_DIR, "models", "tournament")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# The 7 Strategy Combinations
STRATEGIES = {
    "M1_Pure": ["EURUSD_M1.csv"],
    "Intraday_Mix": ["EURUSD_M1.csv", "EURUSD_M5.csv", "EURUSD_M15.csv"],
    "H1_Trend_Master": ["EURUSD_H1.csv"],
    "Multi_TF_Aggressor": ["EURUSD_M1.csv", "EURUSD_M15.csv", "EURUSD_H1.csv", "EURUSD_H4.csv"],
    "Swing_High_TF": ["EURUSD_H4.csv", "EURUSD_D1.csv"],
    "Scalp_Low_TF": ["EURUSD_M1.csv", "EURUSD_M2.csv", "EURUSD_M5.csv"],
    "The_Full_Stack": ["EURUSD_M1.csv", "EURUSD_M5.csv", "EURUSD_M15.csv", "EURUSD_H1.csv", "EURUSD_H4.csv", "EURUSD_D1.csv"]
}

def load_and_sync_safe(files):
    """
    Enhanced load with 'Anti-Cheat' logic.
    Ensures higher timeframes are shifted to avoid looking into the future.
    """
    dfs = []
    primary_tf = files[0] # The timeframe we are actually trading (e.g., M1)
    
    for f in files:
        file_path = os.path.join(DATA_DIR, f)
        if not os.path.exists(file_path):
            continue
            
        tmp = pd.read_csv(file_path)
        tmp.columns = [c.lower() for c in tmp.columns]
        tmp['time'] = pd.to_datetime(tmp['time'])
        tmp = tmp.set_index('time')
        
        suffix = f.split('_')[1].replace('.csv', '').lower()
        
        # --- ANTI-CHEAT PROTOCOL ---
        # 1. Shift non-primary timeframes to ensure zero look-ahead bias
        if f != primary_tf:
            # We shift the data so the model only sees COMPLETED candles
            tmp = tmp.shift(1)
            
        tmp = tmp.add_suffix(f'_{suffix}')
        dfs.append(tmp)
    
    if not dfs: return None, None

    # Sync all timeframes using ffill for different TF granularities
    final_df = pd.concat(dfs, axis=1).fillna(method='ffill').dropna()
    
    # Identify the primary target
    primary_suffix = primary_tf.split('_')[1].replace('.csv', '').lower()
    target_col = f'target_{primary_suffix}'
    
    if target_col in final_df.columns:
        y = final_df[target_col]
        # 2. PURGE ALL TARGETS: Ensure no 'future' labels exist in the features X
        target_cols = [c for c in final_df.columns if 'target' in c]
        X = final_df.drop(target_cols, axis=1)
        
        return X, y
    
    return None, None

results = []
print("--- STARTING VALIDATED ANTI-CHEAT TOURNAMENT ---")

for name, files in STRATEGIES.items():
    print(f"[>] Validating {name}...", end=" ", flush=True)
    X, y = load_and_sync_safe(files)
    
    if X is not None and not X.empty:
        # 80/20 Time-series split
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # XGBoost with slight regularization to prevent overfitting
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.05, 
            eval_metric='logloss',
            reg_alpha=0.1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        results.append({"Strategy": name, "Accuracy": round(acc, 4)})
        print(f"DONE. Real Acc: {acc:.4f}")
        
        # 3. FEATURE IMPORTANCE: Check if one feature is dominating (sign of leakage)
        if acc > 0.85:
            importances = pd.Series(model.feature_importances_, index=X.columns)
            print(f"    [!] High Accuracy Warning for {name}. Top Features:")
            print(importances.sort_values(ascending=False).head(3))
            
    else:
        print("SKIPPED (Check data_vault)")

# --- FINAL STANDINGS & CHARTING ---
if results:
    res_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print("\n--- FINAL STANDINGS (LEAKAGE-FREE) ---")
    print(res_df)
    
    plt.figure(figsize=(10, 6))
    plt.bar(res_df['Strategy'], res_df['Accuracy'], color='darkslateblue')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.title("Validated Tournament Results (Shift-Aligned)")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(BASE_DIR, "tournament_results_chart.png"))
    print(f"\n[+] Standings chart saved to: tournament_results_chart.png")