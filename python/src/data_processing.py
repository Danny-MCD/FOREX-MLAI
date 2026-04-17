import pandas as pd
import numpy as np
import os

# --- SETTINGS ---
# Locates the data_vault relative to the script position
# (Assumes script is in python/src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data_vault")

def add_indicators(df):
    """
    Adds core technical features and the prediction target.
    The Tournament script relies on these columns existing for every timeframe.
    """
    # Ensure columns are lowercase for consistency
    df.columns = [c.lower() for c in df.columns]
    
    # Sort by time to ensure rolling calculations are correct
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
    
    # 1. Trend Features
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # 2. Volatility & Momentum
    df['returns'] = df['close'].pct_change()
    df['range'] = df['high'] - df['low']
    
    # Calculate True Range for ATR
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # 3. THE TARGET (MANDATORY FOR TOURNAMENT)
    # Define target: 1 if the NEXT candle's close is higher than current close, else 0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop rows with NaN values created by rolling windows/shifts
    return df.dropna()

def process_vault():
    """Iterates through all CSV files in the data_vault and prepares them."""
    if not os.path.exists(DATA_DIR):
        print(f"[!] Error: {DATA_DIR} not found. Ensure your data_downloader.py has run.")
        return

    # List all CSVs in the vault
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not files:
        print(f"[!] No CSV files found in {DATA_DIR}.")
        return

    print(f"[*] Found {len(files)} timeframes. Initializing feature engineering...")

    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            # Load the raw data
            df = pd.read_csv(file_path)
            
            # Verify OHLC requirements
            required = ['open', 'high', 'low', 'close']
            if not all(col in [c.lower() for c in df.columns] for col in required):
                print(f" [!] Skipping {filename}: Missing OHLC data columns.")
                continue
            
            # Apply feature engineering and target generation
            processed_df = add_indicators(df)
            
            # Save back to the vault
            processed_df.to_csv(file_path, index=False)
            print(f" [+] Processed {filename}: Added indicators and 'target' column.")
            
        except Exception as e:
            print(f" [!] Error processing {filename}: {e}")

if __name__ == "__main__":
    process_vault()
    print("\n--- PRE-TOURNAMENT PROCESSING COMPLETE ---")
    print("[*] All timeframes are now ready for 'model_tournament.py'.")