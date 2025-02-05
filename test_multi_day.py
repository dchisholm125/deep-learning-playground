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


import pandas as pd

def copy_csv_to_temp_folder(ticker):
    # Read the CSV file
    df = pd.read_csv(f"./csv/{ticker}_data.csv")

    # Save the DataFrame to a new CSV file with a new name
    df.to_csv(f"./temp-csv/temp_csv_{ticker}.csv", index=False)

def train_model_on_temp_csv():
    return True

def make_multi_day_prediction():
    """Add prediction line to `multi_day_predict_{ticker}_{todays_date}.txt` as part of this function, also"""
    return True

def create_df_row(price, close, high, low, open_price, volume):
    return pd.DataFrame({"Price": [price], "Close": [close], "High": [high], "Low": [low], "Open": [open_price], "Volume": [volume]})

def add_prediction_to_temp_csv(ticker):
    df = pd.read_csv(f"./temp-csv/temp_csv_{ticker}.csv")
    df = pd.concat([df,create_df_row(1,2,3,4,5,6)], ignore_index=True)
    df.to_csv(f"./temp-csv/temp_csv_{ticker}.csv", index=False)

copy_csv_to_temp_folder('SPY')
add_prediction_to_temp_csv('SPY')