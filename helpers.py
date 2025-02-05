#!/usr/bin/env python3

# the goal of this file is just the loading the most up-to-date historical CSV file of an array of stocks for further analysis
import pandas as pd
import numpy as np
import requests 
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

class MultiModelPrediction:
    def __init__(self, min_vol_predict, max_vol_predict, min_low_price, avg_low_price, avg_high_price, max_high_price, min_close_predict, max_close_predict):
        self.min_vol_predict = min_vol_predict
        self.max_vol_predict = max_vol_predict
        self.min_low_price = min_low_price
        self.avg_low_price = avg_low_price
        self.avg_high_price = avg_high_price
        self.max_high_price = max_high_price
        self.min_close_predict = min_close_predict
        self.max_close_predict = max_close_predict

### model training ###
def train_model(ticker, loops, csv_file_path = None):
    df_combined = get_model_ready_dataframe(ticker, csv_file_path)

    print(f"From train_model() => using csv_file_path = {csv_file_path}")
    print(df_combined)

    # build stock model and feature sets
    stock_model_1 = RandomForestRegressor(random_state=1)
    stock_features_1 = ['Close_prev', 'High_prev', 'Low_prev', 'Open_prev']
    ### TARGET =  VOLUME ###

    stock_model_2 = RandomForestRegressor(random_state=1)
    stock_features_2 = ['Close_prev', 'Open_prev', 'Volume_prev'] # can share the feature set, no problem!
    ### TARGET =  LOW ###

    stock_model_3 = RandomForestRegressor(random_state=1)
    ### TARGET =  HIGH  ###

    stock_model_4 = RandomForestRegressor(random_state=1)
    stock_features_4 = ['Close_prev', 'High_prev', 'Low_prev', 'Open_prev', 'Volume_prev']
    ### TARGET =  CLOSE  ###

    index_offset = last_index_by_time()
    recent_closed = df_combined.iloc[len(df_combined)-index_offset]

    # establish X (rows to analyze) and y (value to predict) variables
    X1 = df_combined.iloc[1:][stock_features_1]
    recent_closed_X1 = recent_closed[stock_features_1]
    y1 = df_combined.iloc[1:]['Current Volume']

    X2 = df_combined.iloc[1:][stock_features_2] # same feature set
    y2 = df_combined.iloc[1:]['Current Low']

    X3 = df_combined.iloc[1:][stock_features_2] # same feature set
    y3 = df_combined.iloc[1:]['Current High']

    X4 = df_combined.iloc[1:][stock_features_4]
    y4 = df_combined.iloc[1:]['Current Close']

    # start with blank array to insert volume predictions
    volume_predict_arr = []

    # let's start the 1000 random prediction loops here:
    # (this is not "training" the model, we are merely producing a sample of data to derive our educated guesses from)
    print("Start training model...")
    for i in range(loops):
        print(f"Started loop {i} of {loops}")
        # split the training set on each loop
        train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y1)

        # fit first model on TRAINING data set, we want as many random configurations of data points analyzed as possible outcomes, hence the looping
        # from there, we'll take an average --- (and maybe throw out outliers? we may want to take an average of outcomes that are within 1 standard deviation from the mean)
        stock_model_1.fit(train_X1,train_y1)

        prediction_1 = stock_model_1.predict([recent_closed_X1])
        volume_predict_arr.append(prediction_1[0]) # make a prediction and push it to the array

    print(f"End training loops for {ticker}")

    # remove outliers, and keep data to within one standard deviation
    ### removed it , maybe put it back?!?!   # volume_predict_arr = mean_within_one_std(volume_predict_arr)

    # at this point, we can make 'X' predictions and do the same with low, high, and closing prices!
    # let's collect predictions for all elements in the volume_predict_arr

    low_price_arr = []
    high_price_arr = []
    close_predict_arr = []

    # fit last three models on WHOLE data set
    trained_model_2 = stock_model_2.fit(X2,y2) # Low model
    trained_model_3 = stock_model_3.fit(X3,y3) # high model
    trained_model_4 = stock_model_4.fit(X4,y4) # close model


    for volume_entry in volume_predict_arr:
        # format to DataFrame that stock models 2 & 3 expect
        low_high_df = pd.DataFrame({'Close_prev': [recent_closed["Close_prev"]], 'Open_prev': [recent_closed["Open_prev"]], 'Volume_prev': [volume_entry], })

        # make low and high predictions
        low_predict = stock_model_2.predict(low_high_df)[0]
        high_predict = stock_model_3.predict(low_high_df)[0]

        low_price_arr.append(low_predict)
        high_price_arr.append(high_predict)

        # now we can make closing price predictions
        close_predict_df = pd.DataFrame({'Close_prev': [recent_closed["Close_prev"]], 'High_prev': [high_predict], 'Low_prev': [low_predict], 
                        'Open_prev': [recent_closed["Open_prev"]], 'Volume_prev': [volume_entry], })
        
        close_predict = stock_model_4.predict(close_predict_df)[0]

        close_predict_arr.append(close_predict)

    # return the results as an object: the trained model!! 
    model = MultiModelPrediction(min(volume_predict_arr), max(volume_predict_arr), min(low_price_arr), np.mean(low_price_arr), 
                                  np.mean(high_price_arr), max(high_price_arr), min(close_predict_arr), max(close_predict_arr))       

    print(f"From train_model() => model = {model}")

    return model


### Load CSV data helpers ###

def load_csv_files(tickers):
    for ticker in tickers:
        load_data(ticker)

        now = datetime.now()
        timestamp = now.strftime("%H:%M %m/%d/%Y")

        print(f"Loading historical data for '{ticker}' as of {timestamp}")

def load_data(ticker, csv_file_path = None):
    
    # if we're designating a csv, no need to initiate downloads
    if csv_file_path != None:
        return pd.read_csv(csv_file_path)
    else:

        unsafe_session = requests.session()
        unsafe_session.verify = False

        yf.download(tickers=ticker
                    , session=unsafe_session
                    ).to_csv(f'./csv/{ticker}_data.csv')

        # load data into DataFrame
        return pd.read_csv(f'./csv/{ticker}_data.csv')


### Model accuracy helpers ###

def last_index_by_time():
    # get the data row that corresponds with most recent closed price
    # At or before 8:59am, use last line
    # Between 9:00am and 4:00pm, use second to last line
    # At or later than 4:01pm on a market day, use last line

    now = datetime.now()
    return 2 if 9 <= ((now.hour + now.minute) / 100) <= 16 else 1

def get_last_training_row(ticker, csv_file = None):
    # load data into a DataFrame
    read_csv_file = csv_file if csv_file != None else f"./csv/{ticker}_data.csv"

    print(f"read_csv_file = '{read_csv_file}'")
    df = pd.read_csv(read_csv_file)

    # grab index_offset based on time of program running
    index_offset = last_index_by_time()

    return df.loc[len(df)-index_offset]

def get_last_close_price(ticker, csv_file = None):
    # needs to be converted to_numeric
    return pd.to_numeric(get_last_training_row(ticker, csv_file)['Close'], errors='coerce')

def get_price_error(ticker, actual_close):
    return actual_close - get_last_close_price(ticker)

def get_percent_error(ticker, actual_close):
    return f"{get_price_error(ticker, actual_close) / get_last_close_price(ticker) * 100}%"

def get_min_max(value_1, value_2):
    return [value_1, value_2] if value_1 <= value_2 else [value_2, value_1]


### Array Math helpers ###

def mean_within_one_std(arr):

    mean = np.mean(arr)
    std = np.std(arr)

    filtered_arr = [x for x in arr if mean - std <= x <= mean + std]

    return filtered_arr


### File Writing helpers ###
def add_line_to_file(file_path, new_line):

    with open(file_path, "a") as file:
        file.write(f"{new_line}\n")

def add_lines_to_file(file_path, new_lines_arr):

    for new_line in new_lines_arr:
        add_line_to_file(file_path, new_line)

def multi_model_prediction_logger(ticker, loops, multi_model_obj, file_name):

    now = datetime.now()
    timestamp = now.strftime("%H:%M %m/%d/%Y")
    
    last_close = get_last_close_price(ticker)

    arr = [ 
        "-------------------------------------------------------------------------------\n",
        f"FOR TICKER \n\n\t'{ticker}'\n\n",
        f"Prediction generated at: {timestamp}\n\n"
        f"\t\tThe predicted volume range is: {multi_model_obj.min_vol_predict} - {multi_model_obj.max_vol_predict}",
        f"\t\tMin Predicted Low point for the day is: {multi_model_obj.min_low_price}",
        f"\t\tAverage Predicted Low point for the day is: {multi_model_obj.avg_low_price}",
        f"\t\tAverage Predicted High point for the day is: {multi_model_obj.avg_high_price}",
        f"\t\tMax Predicted High point for the day is: {multi_model_obj.max_high_price}",
        f"Close price range for next trading day on '{ticker}' is: {multi_model_obj.min_close_predict} - {multi_model_obj.max_close_predict}\n\n",
        f"Last close was: {last_close}\n\n",
        f"Predictions are based on {loops} training loops:\n\n",
        f"This means the model predicts a difference of {multi_model_obj.min_close_predict - last_close} - {multi_model_obj.max_close_predict - last_close}\n",
        f"And a percentage change of {get_percent_error(ticker, multi_model_obj.min_close_predict)} - {get_percent_error(ticker, multi_model_obj.max_close_predict)}\n",
        "\n\n-------------------------------------------------------------------------------\n"
    ]

    add_lines_to_file(f"{file_name}.txt", arr)

def get_model_ready_dataframe(ticker, csv_file_path = None):
    # get only data rows by 
    data_rows_only = load_data(ticker, csv_file_path)
    
    now = datetime.now()
    current_day = now.strftime("%Y-%m-%d")
    last_index = 1 if current_day == data_rows_only.loc[len(data_rows_only)-1]['Price'] and now.hour < 16 else 0 # market closes at 4 (the 16th hour of the day) don't use today's info until after market close

    if last_index == 1:
        print("We're using yesterday's data")
    else:
        print("Market closed already, take current day's data.")
    
    data_rows_only = data_rows_only.iloc[2:len(data_rows_only)-last_index]

    #shift data, concat() columns, and rename for analyzing data
    df_shifted = data_rows_only.shift(1)
    df_shifted.columns = ['Price_prev', 'Close_prev', 'High_prev', 'Low_prev', 'Open_prev', 'Volume_prev']
    df_combined = pd.concat([data_rows_only.loc[:,], df_shifted], axis=1)
    df_combined.columns = ['Current Date', 'Current Close', 'Current High', 'Current Low', 'Current Open', 'Current Volume',
                        'Day_prev', 'Close_prev', 'High_prev', 'Low_prev', 'Open_prev', 'Volume_prev']
    
    return df_combined
