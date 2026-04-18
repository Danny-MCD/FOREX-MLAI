import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import os

# --- PATH LOGIC ---
# This finds the 'FOREX-MLAI' root folder automatically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up to root, then down into the vault
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
SAVE_DIR = os.path.join(ROOT_DIR, "python", "data_vault", "historical_years")

# Create the folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# --- SETTINGS ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
YEARS = [2022, 2023, 2024, 2025]

def get_data():
    if not mt5.initialize():
        print(f"[-] MT5 Init failed: {mt5.last_error()}")
        return

    print(f"[+] Root Directory detected: {ROOT_DIR}")
    print(f"[+] Saving data to: {SAVE_DIR}")
    
    utc_tz = pytz.timezone("Etc/UTC")

    for year in YEARS:
        print(f"[*] Fetching {year}...")
        start_date = datetime(year, 1, 1, tzinfo=utc_tz)
        end_date = datetime(year, 12, 31, 23, 59, tzinfo=utc_tz)
        
        # copy_rates_range is more robust than copy_rates_from
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)
        
        if rates is None or len(rates) <= 1:
            print(f"[-] Only found {len(rates) if rates is not None else 0} bars for {year}.")
            print("    Check if MT5 is still downloading or if 'Max bars in chart' is set to Unlimited.")
            continue
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Matches your required columns for backtesting
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        
        file_name = f"EURUSD_{year}.csv"
        full_path = os.path.join(SAVE_DIR, file_name)
        
        df.to_csv(full_path, index=False)
        print(f"[+] SUCCESS: {len(df)} bars saved to {file_name}")

    mt5.shutdown()

if __name__ == "__main__":
    get_data()