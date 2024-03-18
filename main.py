# Yahoo finance API
import yfinance as yf
import pandas as pd
import os
# Random forest classifier model
from sklearn.ensemble import RandomForestClassifier
# Used to calculate accuracy of predictor
from sklearn.metrics import precision_score

# Initialize ticker class for S&P 500 index
sp500 = yf.Ticker('^GSPC')

# Query historical prices from a CSV file if exists, otherwise fetch from Yahoo Finance API
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

# Removing irrelevant columns
del sp500['Dividends']
del sp500['Stock Splits']

# Creating a target variable based on tomorrow's price compared to today's price
sp500['Tomorrow'] = sp500['Close'].shift(-1)
sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)

# Trimming data to start from 1990-01-01
sp500 = sp500.loc['1990-01-01':].copy()

# Initializing Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1, n_jobs=-1)

# Splitting data into training and testing sets
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ['Close', 'Volume', 'Open', 'High', 'Low']

# Training the model
model.fit(train[predictors], train['Target'])

# Making predictions on the test set
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Perform backtesting
predictions = backtest(sp500, model, predictors)

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]

# Remove rows with missing data
sp500 = sp500.dropna()

# Adjusting the model parameters and predictions
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1, n_jobs=-1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    # Applying a custom threshold to increase accuracy
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Printing evaluation metrics
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts() / predictions.shape[0])
print(predictions)
