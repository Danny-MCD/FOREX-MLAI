import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_vault", "processed_variants")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train_scalper():
    """Trains an XGBoost model optimized for fast-paced scalping."""
    print("\n[!] Training SCALPING Model (XGBoost)...")
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset_scalping.csv"))
    
    # Features: Volatility, Wicks, and VWAP distance
    features = ['vwap', 'std_dev', 'body_size', 'upper_wick', 'lower_wick', 'dist_sma_20', 'is_London', 'is_NewYork']
    X = df[features]
    y = df['target']
    
    # Time-series split (no shuffle)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # XGBoost is better at picking up small price action nuances
    model = XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Scalper Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    joblib.dump(model, os.path.join(MODEL_DIR, "model_scalper_v1.pkl"))
    print(f"[+] Scalper saved to: models/model_scalper_v1.pkl")

def train_trend_follower():
    """Trains a Random Forest model for higher-TF trend confirmation."""
    print("\n[!] Training TREND Model (Random Forest)...")
    df = pd.read_csv(os.path.join(DATA_DIR, "dataset_trend.csv"))
    
    # Features: SMAs, H1 Trend, and RSI
    features = ['sma_20', 'sma_50', 'h1_trend', 'rsi', 'day_of_week']
    X = df[features]
    y = df['target']
    
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Random Forest is more robust for general trend structure
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Trend Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    joblib.dump(model, os.path.join(MODEL_DIR, "model_trend_v1.pkl"))
    print(f"[+] Trend Model saved to: models/model_trend_v1.pkl")

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    train_scalper()
    train_trend_follower()