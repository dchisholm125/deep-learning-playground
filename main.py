from datetime import datetime
from helpers import add_features_to_CSVs, add_lagged_features_to_CSVs, train_single_model

tickers = [
    # 'AAPL',
    # 'AMZN',
    # 'INTC',
    # 'MSFT',
    # 'QQQ',
    # 'SH',
    'SPY',
    # 'TEAM',
    # 'XYLD'
]

# load all tickers CSV files for today
add_features_to_CSVs(tickers)
add_lagged_features_to_CSVs(tickers, 30)

train_single_model('SPY', 14, 'Close')

# now we're done pre-processing, let's train!

# # prediction loops
# for ticker in tickers:
#     loops = 100

#     trained_model = helpers.train_model(ticker, loops)
#     helpers.multi_model_prediction_logger(ticker, loops, trained_model, "predictions_for_2_5_2025")
