//+------------------------------------------------------------------+
//|                                                 AI_Trader_v1.mq5 |
//+------------------------------------------------------------------+
#property copyright "Danny-MCD"
#property strict

// This EA will act as the executor. 
// For now, let's set it up to display the AI status on the chart.

int OnInit()
{
   EventSetTimer(5); // Check for signals every 5 seconds
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   // Try to open the file
   // The FILE_COMMON flag is the key here!
   int handle = FileOpen("ai_signal.txt", FILE_READ|FILE_TXT|FILE_COMMON);
   
   if(handle != INVALID_HANDLE)
   {
      string signal = FileReadString(handle);
      FileClose(handle);
      
      string status = (signal == "1") ? "BUY SIGNAL ACTIVE" : "WAITING...";
      
      Comment("AI ROBOT STATUS:\n" +
              "Connection: ACTIVE\n" + 
              "Current Signal: " + status + "\n" +
              "Last Update: " + TimeToString(TimeCurrent()));
   }
   else
   {
      // If the file can't be found, show an error on the chart
      Comment("AI ROBOT STATUS:\n" +
              "Connection: FILE NOT FOUND\n" +
              "Looking in: MQL5/Files/ai_signal.txt\n" +
              "Error Code: " + (string)GetLastError());
   }
}

void OnTick()
{
   // Future logic: If Python signals BUY, check if we have a position open
}