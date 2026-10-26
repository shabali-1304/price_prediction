# Cryptocurrency or Stock Price Trend Analysis and Prediction

Beginner-friendly regression workflow for predicting the next day's closing price from historical OHLCV data. It includes data-quality checks, returns, moving averages, volatility, chronological splits, three regressors, a naive baseline, charts, metrics, and model saving.

> **Important:** Educational use only. This is not financial advice, not a trading recommendation, and does not guarantee profit.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data data/sample_prices.csv
```

Generated charts, metrics, predictions, and the best model appear in `outputs/`.

## Use another CSV or download history

Your CSV needs `Date` and `Close`; `Open`, `High`, `Low`, and `Volume` are optional. Run `python src/train.py --data path/to/prices.csv`.

For optional yfinance download support: `pip install yfinance`, then run `python src/train.py --download BTC-USD --period 5y` (stocks such as `AAPL` also work).

## Pipeline

The script sorts by date, reports missing/duplicate records, removes unusable rows, creates daily returns, 7/21-day moving averages, rolling 7-day volatility, and a next-day close target. It uses a chronological 60/20/20 train/validation/test split with no shuffling, compares a naive today-is-tomorrow baseline with Linear Regression, Decision Tree, and Random Forest, reports MAE/MSE/RMSE/R², selects the lowest validation RMSE model, and saves it as `outputs/best_model.joblib`.

Financial prices are noisy; validation results can change over time and lower prediction error does not imply profitability. Transaction costs, taxes, liquidity, market regime changes, and news are outside this beginner example.

## Web dashboard

Install the additional dashboard dependency and start the browser app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser window will open. Choose the included sample dataset or upload your own CSV, then click **Run analysis**. The dashboard displays the selected model, MAE/MSE/RMSE/R², charts, and download buttons for predictions, metrics, and the saved model.
