import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Get user input
ticker = input("Enter stock ticker: ")
print(f"\nFetching data for {ticker}...")

# Pull data
stock = yf.Ticker(ticker)
history_1y = stock.history(period="1y")
history_5y = stock.history(period="5y")

# Current price
current_price = history_1y["Close"].iloc[-1]
print(f"Current Price: ${current_price:.2f}")



# Technical indicators
ma_50 = history_1y["Close"].rolling(window=50).mean().iloc[-1]
ma_200 = history_1y["Close"].rolling(window=200).mean().iloc[-1]
volatility_daily = history_1y["Close"].pct_change().std()
volatility_annual = volatility_daily * np.sqrt(252)

# Moving average signal
if current_price > ma_50 and current_price > ma_200:
    ma_signal = "BULLISH"
elif current_price < ma_50 and current_price < ma_200:
    ma_signal = "BEARISH"
else:
    ma_signal = "NEUTRAL"

# Annual return
start_price = history_1y["Close"].iloc[0]
annual_return = ((current_price - start_price) / start_price) * 100

print(f"50 Day MA:      ${ma_50:.2f}")
print(f"200 Day MA:     ${ma_200:.2f}")
print(f"Annual Return:  {annual_return:.2f}%")
print(f"Volatility:     {volatility_annual:.2%}")
print(f"MA Signal:      {ma_signal}")


# Machine learning prediction
df = history_5y.copy()
df["Return"] = df["Close"].pct_change()
df["MA_50"] = df["Close"].rolling(window=50).mean()
df["MA_200"] = df["Close"].rolling(window=200).mean()
df["Volatility"] = df["Return"].rolling(window=20).std()
df["Return_2"] = df["Return"].shift(1)
df["Return_3"] = df["Return"].shift(2)
df["Volume_Change"] = df["Volume"].pct_change()
df["Momentum"] = df["Close"] - df["Close"].shift(10)
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

features = ["Return", "MA_50", "MA_200", "Volatility", "Return_2", "Return_3", "Volume_Change", "Momentum"]
X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

latest = df[features].iloc[-1].values.reshape(1, -1)
ml_prediction = model.predict(latest)[0]
ml_signal = "BULLISH" if ml_prediction == 1 else "BEARISH"

print(f"ML Prediction:  {ml_signal}")


# Combined signal
signals = [ma_signal, ml_signal]
bullish_count = signals.count("BULLISH")
bearish_count = signals.count("BEARISH")

if bullish_count > bearish_count:
    overall_signal = "BUY"
elif bearish_count > bullish_count:
    overall_signal = "SELL"
else:
    overall_signal = "HOLD"

# Options recommendation
# Strike price based on expected move using volatility
T = 30 / 365
expected_move = current_price * volatility_annual * np.sqrt(T)

if overall_signal == "BUY":
    option_type = "CALL"
    suggested_strike = round(current_price + (expected_move * 0.5), 2)
elif overall_signal == "SELL":
    option_type = "PUT"
    suggested_strike = round(current_price - (expected_move * 0.5), 2)
else:
    option_type = "WAIT - No clear signal"
    suggested_strike = round(current_price, 2)

# Black-Scholes pricing
risk_free_rate = 0.05
d1 = (np.log(current_price / suggested_strike) + (risk_free_rate + 0.5 * volatility_annual**2) * T) / (volatility_annual * np.sqrt(T))
d2 = d1 - volatility_annual * np.sqrt(T)
call_price = current_price * norm.cdf(d1) - suggested_strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
put_price = suggested_strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - current_price * norm.cdf(-d1)

print(f"\n--- OVERALL ANALYSIS ---")
print(f"Overall Signal:    {overall_signal}")
print(f"Option Type:       {option_type}")
print(f"Suggested Strike:  ${suggested_strike}")
print(f"Suggested Expiry:  30 days")
print(f"Fair Call Price:   ${call_price:.2f}")
print(f"Fair Put Price:    ${put_price:.2f}")


# Chart
plt.figure(figsize=(12, 6))
plt.plot(history_1y.index, history_1y["Close"], label="Price", color="blue")
plt.axhline(y=ma_50, color="orange", linestyle="--", label=f"50 Day MA ${ma_50:.2f}")
plt.axhline(y=ma_200, color="red", linestyle="--", label=f"200 Day MA ${ma_200:.2f}")
plt.axhline(y=suggested_strike, color="green", linestyle="--", label=f"Suggested Strike ${suggested_strike}")
plt.title(f"{ticker} - Market Analysis")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()