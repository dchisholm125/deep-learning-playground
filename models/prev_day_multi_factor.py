import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

def train_model(ticker, loops):
    df_combined = helpers.get_model_ready_dataframe(ticker)

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

    prediction_array = []

    # let's start the 1000 random prediction loops here:
    # (this is not "training" the model, we are merely producing a sample of data to derive our educated guesses from)

    for i in range(loops):
        # split the training set on each loop
        train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y1)

        # fit first model on TRAINING data set, we want as many random configurations of data points analyzed as possible outcomes, hence the looping
        # from there, we'll take an average --- (and maybe throw out outliers? we may want to take an average of outcomes that are within 1 standard deviation from the mean)
        stock_model_1.fit(train_X1,train_y1)

        prediction_1 = stock_model_1.predict([recent_closed_X1])
        prediction_array.append(prediction_1[0]) # make a prediction and push it to the array

    prediction_array = helpers.mean_within_one_std(prediction_array)

    low_volume = np.min(prediction_array).astype(np.float64)
    high_volume = np.max(prediction_array).astype(np.float64)

    # fit last three models on WHOLE data set
    stock_model_2.fit(X2,y2) # Low model
    stock_model_3.fit(X3,y3) # high model
    stock_model_4.fit(X4,y4) # close model

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

    low_close, high_close = helpers.get_min_max(prediction_4, prediction_5)

    helpers.multi_model_prediction_logger(ticker, low_volume, high_volume, low_price, high_price, low_close, high_close)
