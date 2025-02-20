import pandas as pd
from helper_dir import csv_prep as cp
from helper_dir import predict
from helpers import add_line_to_file

def generate_horizon_predictions(tickers, timestamp, horizon, base_features):
    # run once for every stock ticker
    for ticker in tickers:
        cp.generate_model_consumable_csvs(ticker, timestamp, 14)

        # adds the "base" line on the prediction CSV
        predict.add_prediction_line_to_csv(ticker, timestamp, horizon, base_features, f"./modified-csv/{ticker}_shared_{timestamp}.csv")
        
        # add additional predictions up to horizon times in loop
        for i in range(1, horizon):
            predict.add_prediction_line_to_csv(ticker, timestamp, horizon, base_features, f"./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv")

        # read in the altered CSV as a dataframe
        df = pd.read_csv(f'./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv')

        total_30_day_return = 1 # start as if the stock return is 100% of itself, no gain, no loss

        for return_perc in df['Return']:
            print(f"{return_perc} * total_30_day_return = ")
            total_30_day_return += (return_perc / 100)
            print(total_30_day_return)

        return_str = f"{timestamp} - Total Return over the next {horizon} days is predicted to be = {(total_30_day_return * 100) - 100}%"

        add_line_to_file(f"./predict-files/{ticker}-{horizon}-day-prediction-results.txt", return_str)

def generate_backdated_predictions(days_back, ticker, timestamp, horizon, base_features):
    # get DataFrame from `days_back` and place it in it's own folder
    df = pd.read_csv(f"./modified-csv/{ticker}_shared_{timestamp}.csv")
    df_copy = df[-days_back-1:-days_back].copy()
    file_path = f"./backdate-tests/{ticker}-{days_back}-backdate_test.csv"
    df_copy.to_csv(file_path, index=False)

    # for horizon, make predictions and add them to a csv
    for i in range(1, horizon):
        predict.backdated_prediction_line_add(days_back, 
                                                ticker, 
                                                timestamp, 
                                                horizon, 
                                                base_features, 
                                                file_path)
        
    # read in the altered CSV as a dataframe
    df = pd.read_csv(file_path)

    total_30_day_return = 1 # start as if the stock return is 100% of itself, no gain, no loss

    for return_perc in df['Return']:
        print(f"{return_perc} * total_30_day_return = ")
        total_30_day_return += (return_perc / 100)
        print(total_30_day_return)

    return_str = f"{days_back} days ago - Total Return over the next {horizon} days is predicted to be = {(total_30_day_return * 100) - 100}%"

    add_line_to_file(f"./backdated-files/backdated_{ticker}-{days_back}-day-prediction-results.txt", return_str)

    print("hey HEY hey")