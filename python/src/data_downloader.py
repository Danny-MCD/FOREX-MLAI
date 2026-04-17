import MetaTrader5 as mt5
import pandas as pd
import os
import time

# --- CONFIGURATION ---
SYMBOL = "EURUSD"
BAR_COUNT = 50000 

# ABSOLUTE PATH LOGIC: 
# This finds your 'data_vault' folder correctly even when running from 'src'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # python/src
BASE_DIR = os.path.dirname(SCRIPT_DIR)                 # python/
DATA_DIR = os.path.join(BASE_DIR, "data_vault")         # python/data_vault

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

def initialize_vault():
    """Ensure the directory exists using the calculated absolute path."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[*] Success: Created Data Vault at: {DATA_DIR}")
    else:
        print(f"[*] Path confirmed: {DATA_DIR}")

def download_all_timeframes():
    if not mt5.initialize():
        print("[!] MT5 Init Failed. Is MetaTrader 5 open?")
        return

    # Force MT5 to select the symbol in MarketWatch (Enables data access)
    if not mt5.symbol_select(SYMBOL, True):
        print(f"[!] Error: {SYMBOL} not found in MarketWatch. Check symbol name.")
        mt5.shutdown()
        return

    print(f"\n--- HARVESTING {SYMBOL} ({BAR_COUNT} bars) ---")
    
    for name, tf_val in TIMEFRAMES.items():
        print(f"[>] Requesting {name}...", end=" ", flush=True)
        
        # Initial pull
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_val, 0, BAR_COUNT)
        
        # Retry logic: Give MT5 2 seconds to fetch history from the broker
        if rates is None or len(rates) == 0:
            print("Syncing...", end=" ", flush=True)
            time.sleep(2) 
            rates = mt5.copy_rates_from_pos(SYMBOL, tf_val, 0, BAR_COUNT)

        if rates is None or len(rates) == 0:
            print(f"FAILED. (MT5 Error Code: {mt5.last_error()})")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Save to file using the absolute path
        file_path = os.path.join(DATA_DIR, f"{SYMBOL}_{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"DONE. Saved {len(df)} rows.")

    print("\n--- PHASE 1 COMPLETE: DATA VAULT IS READY ---")
    mt5.shutdown()

if __name__ == "__main__":
    initialize_vault()
    download_all_timeframes()