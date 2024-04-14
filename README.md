# Stock Market Prediction with Random Forest Classifier

This project aims to predict the direction of the S&P 500 index using historical price and volume data. It utilizes machine learning techniques, specifically a Random Forest Classifier, to forecast whether the index will increase or decrease in the future.

## Key Features:
- **Data Collection**: Historical price data for the S&P 500 index is fetched from Yahoo Finance API.
- **Preprocessing**: Irrelevant columns are removed, and a target variable is created based on tomorrow's price compared to today's price.
- **Model Training**: A Random Forest Classifier model is trained using historical data, with features such as closing price, volume, open, high, and low.
- **Backtesting**: The trained model is backtested on historical data to evaluate its performance over different time horizons.
- **Feature Engineering**: Additional features such as rolling averages, ratios, and trend indicators are engineered to enhance model performance.
- **Model Evaluation**: Precision score and other evaluation metrics are calculated to assess the model's accuracy.

## Project Structure:
- `main.py`: Contains the main script for data preprocessing, model training, backtesting, and evaluation.
- `requirements.txt`: Lists all the required dependencies for running the project.

## Usage:
1. Install the required dependencies listed in `requirements.txt`.
2. Run `main.py` to execute the entire pipeline, from data preprocessing to model evaluation.

## Future Improvements:
- Incorporate more advanced machine learning algorithms.
- Optimize model hyperparameters for better performance.
- Explore additional features and data sources for improved prediction accuracy.

## Acknowledgements:
This project utilizes the `yfinance` library for fetching historical stock data and the `scikit-learn` library for machine learning implementations.
