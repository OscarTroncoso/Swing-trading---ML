import json
from datetime import datetime
import yfinance as yf

# Fetch EUR/USD data (1 year for robust historical simulation)
ticker = "EURUSD=X"
data = yf.download(ticker, period="1y", interval="1d", progress=False)

if data.empty:
    print("Error: Could not fetch data from Yahoo Finance.")
    exit(1)

# Flatten columns if multi-index
if hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

dates = data.index.strftime("%Y-%m-%d").tolist()
closes = [float(c) for c in data["Close"].tolist()]

# Technical Parameters
period_bb = 20
period_rsi = 14
period_sma50 = 50

if len(closes) < period_sma50:
    print("Not enough data for SMA 50")
    exit(1)

# Current Indicators
slice_bb = closes[-period_bb:]
sma20 = sum(slice_bb) / period_bb
variance = sum((x - sma20) ** 2 for x in slice_bb) / period_bb
st_dev = variance ** 0.5

upper_band = round(sma20 + (2 * st_dev), 4)
lower_band = round(sma20 - (2 * st_dev), 4)
current_price = closes[-1]
sma50 = sum(closes[-period_sma50:]) / period_sma50

# RSI Calculation function
def calculate_rsi(prices, periods):
    if len(prices) < periods + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(len(prices) - periods, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    
    avg_gain = sum(gains) / periods
    avg_loss = sum(losses) / periods
    
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

rsi = calculate_rsi(closes, period_rsi)

# --- ROBUST EFFICIENCY SIGNAL LOGIC ---
signal = "NEUTRAL (WAIT)"
stop_loss = None
take_profit = None

risk_multiplier = 1.2
reward_multiplier = 1.8

# Optimized thresholds to capture high-probability market swings effectively
if current_price <= lower_band or rsi < 45:
    signal = "BUY (LONG)"
    stop_loss = round(current_price - (st_dev * risk_multiplier), 4)
    take_profit = round(current_price + (st_dev * reward_multiplier), 4)
elif current_price >= upper_band or rsi > 55:
    signal = "SELL (SHORT)"
    stop_loss = round(current_price + (st_dev * risk_multiplier), 4)
    take_profit = round(current_price - (st_dev * reward_multiplier), 4)
else:
    stop_loss = round(current_price - (st_dev * risk_multiplier), 4)
    take_profit = round(current_price + (st_dev * reward_multiplier), 4)

# --- HIGH-EFFICIENCY BACKTESTING MODULE ---
initial_capital = 10000.0
capital = initial_capital
position = 0  
entry_hist_price = 0.0
entry_date = ""
active_sl = 0.0
active_tp = 0.0

trades = []
peak_capital = initial_capital
max_drawdown = 0.0
gross_profit = 0.0
gross_loss = 0.0

for i in range(period_sma50, len(closes)):
    h_slice_bb = closes[i - period_bb : i]
    h_sma20 = sum(h_slice_bb) / period_bb
    h_var = sum((x - h_sma20) ** 2 for x in h_slice_bb) / period_bb
    h_std = h_var ** 0.5
    h_upper = h_sma20 + (2 * h_std)
    h_lower = h_sma20 - (2 * h_std)
    
    p_close = closes[i]
    h_rsi = calculate_rsi(closes[:i+1], period_rsi)

    # EFFICIENT ENTRY LOGIC
    if position == 0:
        if p_close <= h_lower or h_rsi < 45:
            position = 1
            entry_hist_price = p_close
            entry_date = dates[i]
            active_sl = p_close - (h_std * risk_multiplier)
            active_tp = p_close + (h_std * reward_multiplier)
            
        elif p_close >= h_upper or h_rsi > 55:
            position = -1
            entry_hist_price = p_close
            entry_date = dates[i]
            active_sl = p_close + (h_std * risk_multiplier)
            active_tp = p_close - (h_std * reward_multiplier)
            
    # ROBUST EXIT LOGIC
    elif position == 1: 
        if p_close <= active_sl or p_close >= active_tp:
            profit = (p_close - entry_hist_price) * 10000
            capital += profit
            trades.append({
                "entry": entry_date, "exit": dates[i], "type": "BUY",
                "entryPrice": round(entry_hist_price, 4), "exitPrice": round(p_close, 4),
                "profit": round(profit, 2)
            })
            position = 0

    elif position == -1: 
        if p_close >= active_sl or p_close <= active_tp:
            profit = (entry_hist_price - p_close) * 10000
            capital += profit
            trades.append({
                "entry": entry_date, "exit": dates[i], "type": "SELL",
                "entryPrice": round(entry_hist_price, 4), "exitPrice": round(p_close, 4),
                "profit": round(profit, 2)
            })
            position = 0

    if capital > peak_capital:
        peak_capital = capital
    current_dd = (peak_capital - capital) / peak_capital * 100
    if current_dd > max_drawdown:
        max_drawdown = current_dd

for t in trades:
    p_val = t.get("profit", 0)
    if p_val > 0:
        gross_profit += p_val
    else:
        gross_loss += abs(p_val)

winning_trades = [t for t in trades if t.get("profit", 0) > 0]
win_rate = round((len(winning_trades) / len(trades)) * 100, 1) if trades else 0.0
total_return = round(((capital - initial_capital) / initial_capital) * 100, 2)
profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

output = {
    "lastUpdate": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    "currentPrice": round(current_price, 4),
    "dates": dates[-90:], 
    "prices": closes[-90:],
    "upperBand": upper_band,
    "lowerBand": lower_band,
    "sma": round(sma20, 4),
    "sma50": round(sma50, 4),
    "rsi": rsi,
    "signal": signal,
    "stopLoss": stop_loss,
    "takeProfit": take_profit,
    "backtest": {
        "initialCapital": initial_capital,
        "finalCapital": round(capital, 2),
        "totalReturn": total_return,
        "winRate": win_rate,
        "maxDrawdown": round(max_drawdown, 2),
        "profitFactor": profit_factor,
        "totalTrades": len(trades),
        "trades": trades[-10:] if len(trades) >= 10 else trades
    }
}

with open("data.json", "w") as f:
    json.dump(output, f, indent=4)

print("Successfully generated data.json with efficient backtesting parameters.")
