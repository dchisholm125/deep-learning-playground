from datetime import datetime
import pandas as pd
from helper_dir import csv_prep as cp
from helper_dir import model_helpers as mh
from helper_dir import predict

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y")
horizon = 30
ticker = 'SPY'
base_features = ['Close','High','Low','Open','Volume','Return','SMA_50','RSI','MACD','MACD_Signal','MACD_Hist','ATR']

cp.generate_model_consumable_csvs('QQQ', timestamp, 14)

# predict.add_predictions_to_csv(ticker, timestamp, horizon, base_features)

# import os
# import joblib
# import pandas as pd
# import pandas_ta as ta
# from helpers import add_technicals_to_CSVs, add_lagged_features_to_CSVs, train_single_model, make_prediction, replace_col_val_in_csv, copy_last_df_row, move_row

# tickers = [
#     # 'AAPL',
#     # 'AMZN',
#     # 'INTC',
#     # 'MSFT',
#     # 'QQQ',
#     # 'SH',
#     'SPY',
#     # 'TEAM',
#     # 'XYLD'
# ]

# now = datetime.now()
# timestamp = now.strftime("%m-%d-%Y")
# horizon = 30

# # load all tickers CSV files for today
# train_features = add_technicals_to_CSVs(tickers)
# train_features = train_features[1:]

# add_lagged_features_to_CSVs(tickers, horizon)

# print(train_features)

# trained = False
# model = {}
# X = {}

# # run predictions for every ticker in our list
# for ticker in tickers:

#     # make enough rows to match the number of days in horizon
#     for day in range(1, horizon + 1):

#         copy_last_df_row(ticker, horizon, f"./{horizon}-day-prediction-csv/{horizon}-day-{ticker}_asof_{timestamp}.csv")

#         # first, move all "{feature}_lag1..29" to "{feature}_lag2..30"
#         df = pd.read_csv(f"./{horizon}-day-prediction-csv/{horizon}-day-{ticker}_asof_{timestamp}.csv")

#         df = df[len(df)-1:]

#         for feature in train_features:
#             for i in range(horizon-1, 0, -1):
#                 old_field = f"{feature}_lag{i+1}"
#                 new_field = f"{feature}_lag{i}"
#                 print(f"change {feature}_lag{i+1} from {df[old_field].to_list()[0]} >>> what {feature}_lag{i} is: {df[new_field].to_list()[0]}")
#                 replace_col_val_in_csv(ticker, horizon, df[f"{feature}_lag{i}"].to_list()[0], f"{feature}_lag{i+1}", f"./{horizon}-day-prediction-csv/{horizon}-day-{ticker}_asof_{timestamp}.csv")

#             # second, move the current features to "{feature}_lag1"
#             replace_col_val_in_csv(ticker, horizon, df[feature].to_list()[0], f"{feature}_lag1", f"./{horizon}-day-prediction-csv/{horizon}-day-{ticker}_asof_{timestamp}.csv")

#         # third, replace each {feature} with it's prediction, in-place
#         for feature in train_features:
#             # now we're done pre-processing, let's train!
#             # only train ONCE on every feature, store model in a dictionary outside the for loops
#             if(os.path.exists(f'./models/model-{feature}.joblib') and os.path.exists(f'./models/X-{feature}.joblib')):
#                 print("Model found! Skip to predictions!")
#             else:
#                 print('No model found, training!')
#                 model[feature], X[feature] = train_single_model(ticker, 14, feature)
#                 joblib.dump(model[feature], f'./models/model-{feature}.joblib')
#                 joblib.dump(X[feature], f'./models/X-{feature}.joblib')
                
#             prediction = make_prediction(joblib.load(f'./models/model-{feature}.joblib'), joblib.load(f'./models/X-{feature}.joblib'), feature)
#             replace_col_val_in_csv(ticker, horizon, prediction, feature, f"./{horizon}-day-prediction-csv/{horizon}-day-{ticker}_asof_{timestamp}.csv")
        
#         trained = True

#         # at the end of completing each row, add it to the ./modified-csv/ for re-training on the next cycle
#         move_row(df, f"./modified-csv/{ticker}_asof_{timestamp}.csv")
