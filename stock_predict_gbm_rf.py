import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Parameters
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-10-11"

# Download historical data
data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']].dropna()
data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)

# Geometric Brownian Motion Prediction
mu = data['Return'].mean()
sigma = data['Return'].std()
last_price = data['Close'].iloc[-1]
dt = 1/252

np.random.seed(0)
Z = np.random.normal()
gbm_next = last_price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

print(f"GBM predicted next close: {gbm_next:.2f}")

# Machine Learning Approach (Random Forest)
# Feature engineering: use past 5 days' closes as features
lookback = 5
for i in range(1, lookback+1):
    data[f'Close_lag_{i}'] = data['Close'].shift(i)
data.dropna(inplace=True)

features = [f'Close_lag_{i}' for i in range(1, lookback+1)]
X = data[features].values
y = data['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest MSE on test set: {mean_squared_error(y_test, y_pred):.2f}")

# Predict next day's price
latest_features = data[features].iloc[-1].values.reshape(1, -1)
rf_next = rf.predict(latest_features)[0]
print(f"Random Forest predicted next close: {rf_next:.2f}")
