import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Load S&P 500 data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500 = sp500.loc["1990-01-01":].copy()

# Target variable: will it go up tomorrow?
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# ======== Feature Engineering ========= #
# % return
sp500["Return_1d"] = sp500["Close"].pct_change()

# Moving averages
sp500["MA_5"] = sp500["Close"].rolling(window=5).mean()
sp500["MA_10"] = sp500["Close"].rolling(window=10).mean()

# Volatility
sp500["Volatility_5"] = sp500["Return_1d"].rolling(5).std()

# Momentum
sp500["Momentum"] = sp500["Close"] - sp500["Close"].shift(5)

# Drop NA rows from indicators
sp500.dropna(inplace=True)

# Feature set
predictors = ["Close", "Volume", "Open", "High", "Low", 
              "Return_1d", "MA_5", "MA_10", "Volatility_5", "Momentum"]

# Split train/test
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Model
model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
model.fit(train[predictors], train["Target"])

# Predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# ======= Evaluation ======= #
actual = test["Target"]
print("Accuracy:", accuracy_score(actual, preds))
print("Precision:", precision_score(actual, preds))
print("Recall:", recall_score(actual, preds))
print("Confusion Matrix:")
print(confusion_matrix(actual, preds))

# ======= Plot Predictions ======= #
combined = pd.concat([actual, preds], axis=1)
combined.columns = ["Actual", "Predicted"]
combined.plot(figsize=(12, 5), title="S&P 500 Direction: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# ======= Feature Importance ======= #
importances = pd.Series(model.feature_importances_, index=predictors).sort_values()
importances.plot(kind='barh', figsize=(8, 5), title="Feature Importances")
plt.tight_layout()
plt.show()

# ======= (Optional) Simple Strategy Backtest ======= #
test = test.copy()  # To avoid SettingWithCopyWarning
test["Predicted_Signal"] = preds
test["Daily_Return"] = test["Close"].pct_change()
test["Strategy_Return"] = test["Daily_Return"] * test["Predicted_Signal"].shift(1)

test[["Daily_Return", "Strategy_Return"]].cumsum().plot(
    figsize=(10, 5), title="Cumulative Returns: Strategy vs Market")
plt.grid(True)
plt.tight_layout()
plt.show()
