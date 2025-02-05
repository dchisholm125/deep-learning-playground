# If we're doing a multi-day analysis, we need to start our 1st prediction from the ORIGINAL csv data
# all subsequent predictions will require us to add a CSV row, but we DON'T want to alter the ORIGINAL csv files
# we need to produce a temp_csv_{ticker}.csv file while we're running our program, and delete it when we're done because they're not useful
# this file will be produced on the very first "loop" / prediction!!
#
# the csv row will need to be as below:
#
#           Price,Close,High,Low,Open,Volume
#           2025-02-05,42.42639923095703,42.42850112915039,42.31999969482422,42.380001068115234,157062
#
# Then we can write a workflow like below:
#   1. Copy ticker-specific CSV file from `csv/` into `temp-csv/` and name it `temp_csv_{ticker}.csv`
#   2. Train the model based on current data in `temp_csv_{ticker}.csv`
#   3. Make current prediction for current loop
#   4. Add prediction line to `multi_day_predict_{ticker}_{todays_date}.txt`
#   5. IF there are more loops to go, add current prediction to `temp_csv_{ticker}.csv` and continue looping

import os
from datetime import date
import pandas as pd
from helpers import train_model, add_line_to_file, get_last_close_price

class CSVObj:
    def __init__(self, day, close, high, low, open_price, volume):
        self.day = day
        self.close = close
        self.high = high
        self.low = low
        self.open_price = open_price
        self.volume= volume

def copy_csv_to_temp_folder(ticker):
    # Read the CSV file
    df = pd.read_csv(f"./csv/{ticker}_data.csv")

    # Save the DataFrame to a new CSV file with a new name
    df.to_csv(f"./temp-csv/temp_csv_{ticker}.csv", index=False)

def make_multi_day_prediction(ticker, loops, days):
    """Add prediction line to `multi_day_predict_{ticker}_{todays_date}.txt` as part of this function, also"""
    csv_file_path = f"./temp-csv/temp_csv_{ticker}.csv"

    if os.path.exists(csv_file_path) == False:
        copy_csv_to_temp_folder(ticker)
    
    trained_model_obj = train_model(ticker, loops, csv_file_path)
    print("make_multi_day_prediction() => trained model obj = ")
    print(trained_model_obj)
    model_as_df = create_df_row(ticker, trained_model_obj, csv_file_path, days)

    add_prediction_to_txt_file(ticker, trained_model_obj)

    add_prediction_to_temp_csv(model_as_df, csv_file_path, days)

    days -= 1

    if days > 0:
        make_multi_day_prediction(ticker, loops, days)

def create_df_row(ticker, trained_model_obj, csv_file, days = 1,):
    print('re-working create_df_row... obj is:')
    print(trained_model_obj)

    # open price is just the close of the previous day
    previous_close = get_last_close_price(ticker, csv_file)
    
    return pd.DataFrame({"Price": [days], 
                         "Close": [(trained_model_obj.min_close_predict + trained_model_obj.max_close_predict) / 2], 
                         "High": [trained_model_obj.avg_high_price], 
                         "Low": [trained_model_obj.avg_low_price], 
                         "Open": [previous_close], 
                         "Volume": [(trained_model_obj.min_vol_predict + trained_model_obj.max_vol_predict) / 2]}, index=[0]
                                    # , 
                                    # columns=["Price", "Close", "High","Low", "Open", "Volume"]
                                    )

def add_prediction_to_temp_csv(trained_model_obj, csv_file_path, days):
    # df = pd.read_csv(csv_file_path)
    # print(trained_model_obj.loc[0])
    # df = pd.concat([df,trained_model_obj.loc[0]], ignore_index=True)
    # df.to_csv(csv_file_path, index=False)

    with open(csv_file_path, "a") as file:
        file.write(f"{days},{trained_model_obj.Close[0]},{trained_model_obj.High[0]},{trained_model_obj.Low[0]},{trained_model_obj.Open[0]},{trained_model_obj.Volume[0]}\n")

def add_prediction_to_txt_file(ticker, trained_model_obj):
    today = date.today()
    todays_date = today.strftime("%Y-%m-%d")

    add_line_to_file(f"multi_day_predict_{ticker}_{todays_date}.txt", f"{ticker}: {vars(trained_model_obj)}")

# copy_csv_to_temp_folder('SPY')
make_multi_day_prediction('QQQ',10,30)