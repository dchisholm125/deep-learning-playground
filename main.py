from datetime import datetime
import helpers

tickers = [
    'AAPL',
    'AMZN',
    'INTC',
    'MSFT',
    'QQQ',
    'SH',
    'SPY',
    'TEAM',
    'XYLD'
]

for ticker in tickers:
    helpers.train_model(ticker, 10, "predictions_for_2_5_2025")
