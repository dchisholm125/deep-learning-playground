from datetime import datetime
import pandas as pd
from helper_dir import csv_prep as cp
from helper_dir import model_helpers as mh
from helper_dir import predict
from helper_dir.generator import generate_backdated_predictions, generate_backdated_predictions
from helpers import add_line_to_file, add_lines_to_file

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y")
horizon = 30
tickers = [
    # 'AAPL',
    # 'AMZN',
    # 'INTC',
    # 'VTTSX',
    # 'ARKK',
    # 'AIQ',
    # 'QQQ',
    # 'SPY',
    # 'TEAM',
    # 'QYLD',
    # 'RYLD',
    # 'XYLD',
    # 'MSFT',
    # 'SH',
]
base_features = ['Close','High','Low','Open','Volume','Return','SMA_50','RSI','MACD','MACD_Signal','MACD_Hist','ATR']

# for ticker in tickers:
#     predict.predict_from_X_mdays_ago(30, ticker, timestamp, 'Return', f"./modified-csv/{ticker}_shared_{timestamp}.csv")

# mh.test_model_accuracy('QQQ', 'Return', timestamp)

# generate_horizon_predictions(tickers, timestamp, horizon, base_features)

for i in range(30, 40):
    generate_backdated_predictions(i, 'SPY', timestamp, horizon, base_features)
