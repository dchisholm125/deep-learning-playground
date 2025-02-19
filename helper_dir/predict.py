import joblib
import pandas as pd
from helper_dir import model_helpers as mh
from helper_dir import csv_prep as cp

def make_prediction(ticker, timestamp, target_feature, base_features, csv_with_test_data):
    # load model
    model = joblib.load(f"./models/{ticker}-model-{target_feature}-{timestamp}.joblib")

    # load appropriate csv
    df = pd.read_csv(csv_with_test_data)

    # only look at last row
    df = df[-1:]
    print(f'Price = {df.Price}')
    df = df.drop('Price', axis=1)
    df = df.drop(target_feature, axis=1)

    # make a prediction based on the row of information
    prediction = model.predict(df)

    print(f"Model was asked to make a prediction from the LAST row of information. This yielded one prediction for feature {target_feature}:")
    print(prediction[0])

    # prediction is over, and the {horizon}-day-prediction-csv folder already has a csv in it, let's change the value!

    return prediction[0]

def add_prediction_line_to_csv(ticker, timestamp, horizon, features, csv_with_test_data):

    predict_file_path = f"./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv"

    df = pd.read_csv(predict_file_path)
    df_copy = df[-1:].copy()

    df_copy = move_back_lagged_features_df(df_copy, horizon)

    for feature in features:
        mh.train_single_model(ticker, timestamp, feature)
        prediction = make_prediction(ticker, timestamp, feature, features, csv_with_test_data)

        print(f'BEFORE in-place change: {feature} = {df_copy[feature]}')
        # edit LAST lines in-place
        df_copy[feature] = prediction
        print(f'AFTER in-place change: {feature} = {df_copy[feature]}')

    df_copy['Price'] += 1

    pd.concat([df, df_copy]).to_csv(predict_file_path, index=False)

def move_back_lagged_features_df(df, horizon):
    """
    Moves all lagged features from {feature}_lag1..29 >>> {feature}_lag2..30 in prediction CSV.
    """
    base_features = ['Close','High','Low','Open','Volume','Return','SMA_50','RSI','MACD','MACD_Signal','MACD_Hist','ATR']
    
    for feature in base_features:
        for i in range(horizon-1, 0, -1):
            older_field = f"{feature}_lag{i+1}"
            newer_field = f"{feature}_lag{i}"
            df[older_field] = df[newer_field]
            print(f'Moved {newer_field} into {older_field}')

            # second, move the current features to "{feature}_lag1"
            if (i == 1):
                df[newer_field] = df[feature]
                df[feature] = 0
                print(f'Moved {feature} into {newer_field}')

    return df
