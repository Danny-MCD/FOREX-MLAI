import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import os

# --- SETTINGS ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
YEARS_TO_GET = [2022, 2023, 2024, 2025]
SAVE_DIR = "python/data_vault/historical_years"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_data():
    # 1. Initialize Connection
    if not mt5.initialize():
        print(f"[-] MT5 Initialize failed, error code: {mt5.last_error()}")
        return

    print(f"[+] Connected to MT5. Terminal Info: {mt5.terminal_info()._asdict()}")
    timezone = pytz.timezone("Etc/UTC")

    for year in YEARS_TO_GET:
        print(f"[*] Fetching data for {year}...")
        
        # Define start and end dates
        utc_from = datetime(year, 1, 1, tzinfo=timezone)
        utc_to = datetime(year, 12, 31, 23, 59, tzinfo=timezone)
        
        # 2. Request Rates from MT5
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, utc_from, utc_to)
        
        if rates is None or len(rates) == 0:
            print(f"[-] No data found for {year}. Error: {mt5.last_error()}")
            continue
            
        # 3. Process into DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Clean up column names to match your existing scripts
        # (Keeping only what's necessary to save space)
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        
        # 4. Save to CSV
        file_path = os.path.join(SAVE_DIR, f"EURUSD_{year}.csv")
        df.to_csv(file_path, index=False)
        print(f"[+] Saved {len(df)} bars to {file_path}")

    mt5.shutdown()
    print("\n[!] All downloads complete. You can now run the Multi-Year Stress Test.")

if __name__ == "__main__":
    download_data()