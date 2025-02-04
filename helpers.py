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

### model training ###
def train_model(ticker, loops):
    print("locaation 1")
    df_combined = get_model_ready_dataframe(ticker)

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

    print("locaation 2")
    recent_closed = df_combined.iloc[len(df_combined)-1]

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

    # global scope variables for data extraction
    print("locaation 3")
    prediction_array = []

    # let's start the 1000 random prediction loops here:
    # (this is not "training" the model, we are merely producing a sample of data to derive our educated guesses from)
    print("locaation 4")
    for i in range(loops):
        # split the training set on each loop
        train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y1)

        # fit first model on TRAINING data set, we want as many random configurations of data points analyzed as possible outcomes, hence the looping
        # from there, we'll take an average --- (and maybe throw out outliers? we may want to take an average of outcomes that are within 1 standard deviation from the mean)
        stock_model_1.fit(train_X1,train_y1)

        prediction_1 = stock_model_1.predict([recent_closed_X1])
        prediction_array.append(prediction_1[0]) # make a prediction and push it to the array

    prediction_array = mean_within_one_std(prediction_array)

    low_volume = np.min(prediction_array).astype(np.float64)
    high_volume = np.max(prediction_array).astype(np.float64)

    # fit last three models on WHOLE data set
    stock_model_2.fit(X2,y2) # Low model
    stock_model_3.fit(X3,y3) # high model
    stock_model_4.fit(X4,y4) # close model

    print("locaation 5")
    
    df_2 = pd.DataFrame({'Close_prev': [recent_closed["Close_prev"]], 'Open_prev': [recent_closed["Open_prev"]], 'Volume_prev': [low_volume], })
    df_3 = pd.DataFrame({'Close_prev': [recent_closed["Close_prev"]], 'Open_prev': [recent_closed["Open_prev"]], 'Volume_prev': [high_volume], })

    # make two predictions for the HIGH and LOW volume
    low_price = stock_model_2.predict(df_2)[0]
    high_price = stock_model_3.predict(df_3)[0]

    now = datetime.now()
    timestamp = now.strftime("%H:%M %m/%d/%Y")

    # Now let's make the very LAST prediction based on this new information!
    df_4 = pd.DataFrame({'Close_prev': [recent_closed["Close_prev"]], 'High_prev': [high_price], 'Low_prev': [low_price], 
                        'Open_prev': [recent_closed["Open_prev"]], 'Volume_prev': [low_volume], })

    df_5 = pd.DataFrame({'Close_prev': [recent_closed["Close_prev"]], 'High_prev': [high_price], 'Low_prev': [low_price], 
                        'Open_prev': [recent_closed["Open_prev"]], 'Volume_prev': [high_volume], })

    prediction_4 = stock_model_4.predict(df_4)[0]
    prediction_5 = stock_model_4.predict(df_5)[0]

    low_close, high_close = get_min_max(prediction_4, prediction_5)

    multi_model_prediction_logger(ticker, low_volume, high_volume, low_price, high_price, low_close, high_close)


### Load CSV data helpers ###

def load_csv_files(tickers):
    for ticker in tickers:
        load_data(ticker)

        now = datetime.now()
        timestamp = now.strftime("%H:%M %m/%d/%Y")

        print(f"Loading historical data for '{ticker}' as of {timestamp}")

def load_data(ticker):
    unsafe_session = requests.session()
    unsafe_session.verify = False

    yf.download(tickers=ticker
                , session=unsafe_session
                ).to_csv(f'./csv/{ticker}_data.csv')

    # load data into DataFrame
    return pd.read_csv(f'./csv/{ticker}_data.csv')


### Model accuracy helpers ###

def get_last_close_price(ticker):
    # last close price will ALWAYS be the final line of the CSV

    #load data into a DataFrame
    df = pd.read_csv(f'./csv/{ticker}_data.csv')

    # be very careful when loading data.
    # if the data is loaded while the market is OPEN, the last row is not yesterday's information, but today's
    # it is real-time CSV files 

    #To counteract this, check that the final 'Price' does NOT equal today's date
    current_day = datetime.now().strftime("%Y-%m-%d")
    last_index = 2 if current_day == df.loc[len(df)-1]['Price'] else 1

    return pd.to_numeric(df.loc[len(df)-last_index]['Close'], errors='coerce')

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
        file.write(new_line + "\n")

def add_lines_to_file(file_path, new_lines_arr):

    for new_line in new_lines_arr:
        add_line_to_file(file_path, new_line)

def multi_model_prediction_logger(ticker, low_volume, high_volume, low_price, 
                                  high_price, low_close, high_close):
    
    arr = [ 
        "-------------------------------------------------------------------------------\n",
        f"FOR TICKER \n\n\t'{ticker}'\n",
        f"\t\tThe predicted volume range is: {low_volume} - {high_volume}",
        f"\t\tPredicted Low point for the day is: {low_price}",
        f"\t\tPredicted High point for the day is: {high_price}",
        f"Close price range for next trading day on '{ticker}' is: {low_close} - {high_close}"
        "\n\n-------------------------------------------------------------------------------\n"
    ]

    add_lines_to_file("some_file.txt", arr)

def get_model_ready_dataframe(ticker):
    print('before load_data(ticker).iloc[2:]')
    # get only data rows by 
    data_rows_only = load_data(ticker).iloc[2:]
    print('after load_data(ticker).iloc[2:]')

    #shift data, concat() columns, and rename for analyzing data
    df_shifted = data_rows_only.shift(1)
    df_shifted.columns = ['Price_prev', 'Close_prev', 'High_prev', 'Low_prev', 'Open_prev', 'Volume_prev']
    df_combined = pd.concat([data_rows_only.loc[:,], df_shifted], axis=1)
    df_combined.columns = ['Current Date', 'Current Close', 'Current High', 'Current Low', 'Current Open', 'Current Volume',
                        'Day_prev', 'Close_prev', 'High_prev', 'Low_prev', 'Open_prev', 'Volume_prev']
    
    return df_combined
