from datetime import datetime
import pandas as pd
from helper_dir import csv_prep as cp
from helper_dir import model_helpers as mh
from helper_dir import predict
from helpers import add_line_to_file, add_lines_to_file

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y")
horizon = 30
tickers = [
    'AAPL',
    # 'AMZN',
    # 'INTC',
    # 'MSFT',
    'QQQ',
    # 'SH',
    'SPY',
    # 'TEAM',
    # 'XYLD'
]
base_features = ['Close','High','Low','Open','Volume','Return','SMA_50','RSI','MACD','MACD_Signal','MACD_Hist','ATR']

for ticker in tickers:
    predict.predict_from_X_mdays_ago(30, ticker, timestamp, 'Return', f"./modified-csv/{ticker}_shared_{timestamp}.csv")

# mh.test_model_accuracy('QQQ', 'Return', timestamp)

# for ticker in tickers:
#     cp.generate_model_consumable_csvs(ticker, timestamp, 14)

#     predict.add_prediction_line_to_csv(ticker, timestamp, horizon, base_features, f"./modified-csv/{ticker}_shared_{timestamp}.csv")
#     for i in range(1, horizon):
#         predict.add_prediction_line_to_csv(ticker, timestamp, horizon, base_features, f"./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv")

#     df = pd.read_csv(f'./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv')

#     total_30_day_return = 1

#     for return_perc in df['Return']:
#         print(f"{return_perc} * total_30_day_return = ")
#         total_30_day_return += (return_perc / 100)
#         print(total_30_day_return)

#     return_str = f"{timestamp} - Total Return over the next {horizon} days is predicted to be = {(total_30_day_return * 100) - 100}%"

#     add_line_to_file(f"./predict-files/{ticker}-{horizon}-day-prediction-results.txt", return_str)
