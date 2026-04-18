import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- ABSOLUTE PATH SETUP ---
# Based on your sidebar: C:\Users\HalfDog\REPOS\FOREX-MLAI\models
ROOT_DIR = r"C:\Users\HalfDog\REPOS\FOREX-MLAI"
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_scalper_v1.pkl")
VAULT_DIR = os.path.join(ROOT_DIR, "python", "data_vault", "historical_years")

print(f"[*] Target Model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    # Fallback check: Let's see if it's accidentally inside the python folder
    ALT_PATH = os.path.join(ROOT_DIR, "python", "models", "model_scalper_v1.pkl")
    if os.path.exists(ALT_PATH):
        MODEL_PATH = ALT_PATH
    else:
        print(f"[-] FATAL: Cannot find model at {MODEL_PATH}")
        quit()

def eng_features(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    # Feature Engineering for XGBoost
    df['body_size'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['std_dev'] = df['close'].rolling(20).std()
    df['dist_sma_20'] = df['close'] - df['close'].rolling(20).mean()
    df['vwap'] = (df['close'] * df['tick_volume']).cumsum() / (df['tick_volume'].cumsum() + 1e-9)
    df['is_London'] = df['hour'].between(8, 16).astype(int)
    df['is_NewYork'] = df['hour'].between(13, 21).astype(int)
    return df.dropna()

def run_stress_test():
    print("[+] Loading Model...")
    model = joblib.load(MODEL_PATH)
    model_features = model.get_booster().feature_names
    
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    SPREAD = 0.00007 

    files = sorted([f for f in os.listdir(VAULT_DIR) if f.endswith('.csv')])
    
    for file in files:
        year = file.split('_')[-1].replace('.csv', '')
        print(f"[*] Crunching {year}...")
        
        data = pd.read_csv(os.path.join(VAULT_DIR, file))
        data = eng_features(data)
        
        # XGBoost Predictions
        data['prob_up'] = model.predict_proba(data[model_features])[:, 1]
        data['returns'] = data['close'].pct_change().shift(-1).fillna(0)
        
        # Hybrid Logic (Core vs Buffer)
        def hybrid_logic(row):
            h = row['hour']
            if 11 <= h <= 19:
                return 1 if row['prob_up'] > 0.60 else (-1 if row['prob_up'] < 0.40 else 0)
            elif h in [10, 20]:
                return 1 if row['prob_up'] > 0.65 else (-1 if row['prob_up'] < 0.35 else 0)
            return 0

        data['sig'] = data.apply(hybrid_logic, axis=1)
        data['pnl'] = (data['sig'] * data['returns']) - (np.abs(data['sig']) * SPREAD)
        
        plt.plot(data['time'], data['pnl'].cumsum(), label=f"Year {year}")

    plt.title("Multi-Year Hybrid Strategy Stress Test (2022-2025)", fontsize=16)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_stress_test()