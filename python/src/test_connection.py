import MetaTrader5 as mt5

# 1. Initialize the connection to the MT5 terminal
if not mt5.initialize():
    print(f"Initialize() failed, error code = {mt5.last_error()}")
    quit()

# 2. Check connection status and account info
account_info = mt5.account_info()
if account_info is not None:
    print("--- Connection Successful! ---")
    print(f"Logged into Account: {account_info.login}")
    print(f"Broker/Server: {account_info.server}")
    print(f"Balance: {account_info.balance} {account_info.currency}")
else:
    print("Connected to MT5, but could not retrieve account info.")
    print(f"Error: {mt5.last_error()}")

# 3. Always shut down the connection when finished
mt5.shutdown()