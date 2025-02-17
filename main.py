from datetime import datetime
import pandas as pd
import pandas_ta as ta
from helpers import add_features_to_CSVs, add_lagged_features_to_CSVs, train_single_model_and_predict, replace_prediction_in_csv, add_new_csv_row

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

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y")
ticker = 'SPY'
horizon = 30

# load all tickers CSV files for today
train_features = add_features_to_CSVs(tickers)
train_features = train_features[1:]
add_lagged_features_to_CSVs(tickers, horizon)

# print(train_features)

# testing loops
for i in range(1):

    add_new_csv_row('SPY', horizon)

    # first, move all "{feature}_lag1..29" to "{feature}_lag2..30"
    df = pd.read_csv(f"./{horizon}-day-prediction-csv/{ticker}_asof_{timestamp}.csv")

    for feature in train_features:
        for i in range(horizon-1, 0, -1):
            replace_prediction_in_csv('SPY', horizon, df[f"{feature}_lag{i}"].to_list()[0], f"{feature}_lag{i+1}")

        # second, move the current features to "{feature}_lag1"
        replace_prediction_in_csv('SPY', horizon, df[feature].to_list()[0], f"{feature}_lag1")

    # third, replace each {feature} with it's prediction, in-place
    # for feature in train_features:
    #     # now we're done pre-processing, let's train!
    #     prediction = train_single_model_and_predict('SPY', 14, feature)
    #     replace_prediction_in_csv('SPY', horizon, prediction, feature)
        
# # prediction loops
# for ticker in tickers:
#     loops = 100

#     trained_model = helpers.train_model(ticker, loops)
#     helpers.multi_model_prediction_logger(ticker, loops, trained_model, "predictions_for_2_5_2025")
