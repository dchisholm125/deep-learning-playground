from datetime import datetime
import helpers

tickers = [
    # 'AAPL',
    # 'AMZN',
    # 'INTC',
    # 'MSFT',
    # 'QQQ',
    # 'SH',
    # 'SPY',
    'TEAM',
    'XYLD'
]

for ticker in tickers:
    loops = 100

    trained_model = helpers.train_model(ticker, loops)
    helpers.multi_model_prediction_logger(ticker, loops, trained_model, "predictions_for_2_5_2025")
