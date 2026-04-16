import MetaTrader5 as mt5
import pandas as pd
import joblib
import time
import os
import warnings

# Suppress the "Feature Names" warning to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Define Paths
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PYTHON_DIR, "models", "trading_model_v1.pkl")

# This finds the MT5 Common folder automatically
# It usually ends up in: C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\Common\Files
COMMON_PATH = os.path.join(os.environ['APPDATA'], "MetaQuotes", "Terminal", "Common", "Files")
SIGNAL_PATH = os.path.join(COMMON_PATH, "ai_signal.txt")

# Ensure the folder exists
if not os.path.exists(COMMON_PATH):
    os.makedirs(COMMON_PATH)

# 2. Load the trained brain
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Model not found at {MODEL_PATH}")
    quit()

if not mt5.initialize():
    print("MT5 Init Failed")
    quit()

def get_live_prediction():
    # Fetch data
    rates_2m = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M2, 0, 1)
    rates_15m = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 20)
    rates_4h = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H4, 0, 1)

    if rates_2m is None or rates_15m is None: return 0

    # Calculate Features
    df_15m = pd.DataFrame(rates_15m)
    sma_15m = df_15m['close'].mean()
    current_close = rates_2m[0]['close']
    
    dist_from_sma = current_close - sma_15m
    trend_4h = 1 if rates_4h[0]['close'] > rates_4h[0]['open'] else -1
    vol = rates_2m[0]['tick_volume']
    spread = rates_2m[0]['spread']

    # Make Prediction
    X = [[dist_from_sma, trend_4h, vol, spread]]
    prediction = model.predict(X)[0]
    return prediction

print("--- AI Server Running ---")
try:
    while True:
        pred = get_live_prediction()
        signal = "1" if pred == 1 else "0"
        
        # Write signal for MQL5 to read
        with open(SIGNAL_PATH, "w") as f:
            f.write(signal)
            
        status = "BUY" if pred == 1 else "WAIT"
        print(f"Time: {time.ctime()} | Prediction: {status} | Signal Saved")
        
        time.sleep(10) 
except KeyboardInterrupt:
    print("Server Stopped")
    mt5.shutdown()