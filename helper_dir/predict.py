import joblib
import pandas as pd
from helper_dir import model_helpers as mh
from helper_dir import csv_prep as cp

def make_prediction(ticker, timestamp, target_feature, csv_with_test_data):
    """
    Make prediction based on existing model, CSV provided,
    """
    # load model
    model = joblib.load(f"./models/{ticker}-model-{target_feature}-{timestamp}.joblib")

    # load appropriate csv
    df = pd.read_csv(csv_with_test_data)

    # only look at last row
    df = df[-1:]
    print(f'Price = {df.Price}')
    ignore_features = ['Price','Close','High','Low','Open','Volume','Return','SMA_50','RSI','MACD','MACD_Signal','MACD_Hist','ATR']

    df_columns = []

    for feature in df.columns:
        if feature in ignore_features:
            continue
        else:
            df_columns.append(feature)

    print('Columns are:')
    print(df_columns)

    df = df[df_columns]

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
        prediction = make_prediction(ticker, timestamp, feature, csv_with_test_data)

        print(f'BEFORE in-place change: {feature} = {df_copy[feature]}')
        # edit LAST lines in-place
        df_copy[feature] = prediction
        print(f'AFTER in-place change: {feature} = {df_copy[feature]}')

    df_copy['Price'] += 1

    # before adding predictions to the CSV which will be consumed AGAIN for prediction making, let's "normalize" the predictions
    df_copy = normalize_predictions_by_return_perc(df_copy, df[-1:])

    pd.concat([df, df_copy]).to_csv(predict_file_path, index=False)

def backdated_prediction_line_add(days_back, ticker, timestamp, horizon, features, csv_with_test_data):

    data_file_path = f"./backdate-tests/{ticker}-{days_back}-backdate_test.csv"

    df = pd.read_csv(data_file_path)

    # read in the last line for making a prediction
    df_copy = df[-1:].copy()

    df_copy = move_back_lagged_features_df(df_copy, horizon)

    for feature in features:
        # DON'T train the model here, this should only be run on existing models for accuracy purposes / as a datapoint for making decisions
        prediction = make_prediction(ticker, timestamp, feature, csv_with_test_data)

        print(f'BEFORE in-place change: {feature} = {df_copy[feature]}')
        # edit LAST lines in-place
        df_copy[feature] = prediction
        print(f'AFTER in-place change: {feature} = {df_copy[feature]}')

    df_copy['Price'] = 1

    # before adding predictions to the CSV which will be consumed AGAIN for prediction making, let's "normalize" the predictions
    df_copy = normalize_predictions_by_return_perc(df_copy, df[-1:])

    pd.concat([df, df_copy]).to_csv(data_file_path, index=False)

def normalize_predictions_by_return_perc(df_copy, df_actual):
    # 'Return' is the best predicted variable of the group of base features, let's noramlize based on this prediction
    features = ['Close']

    for feature in features:
        # alter the df_copy feature based on yesterday's info
        print(f'For {feature}')
        print(f"df_actual[feature] ({df_actual[feature]}) * (1 + df_copy['Return']) ({(1 + df_copy['Return'])}) = {df_actual[feature] * (1 + (df_copy['Return'] / 100))}")
        df_copy[feature] = df_actual[feature] * (1 + (df_copy['Return'] / 100))

    df_copy['Open'] = df_actual['Close']

    return df_copy

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

def predict_from_X_mdays_ago(days_back, ticker, timestamp, target_feature, csv_with_test_data):
     # load model
    model = joblib.load(f"./models/{ticker}-model-{target_feature}-{timestamp}.joblib")

    # load appropriate csv
    df = pd.read_csv(csv_with_test_data)

    # only look at row from days_back ago
    df = df[-days_back-1:-days_back]
    print(f'Price = {df.Price}')
    df = df.drop('Price', axis=1)
    days_back_close = df['Close'].iloc[0]
    print(f"'Close' as of {days_back} = {days_back_close}")
    df = df.drop(target_feature, axis=1)

    # make a prediction based on the row of information
    prediction = model.predict(df)

    print(f"Model was asked to make a prediction from {days_back} days ago. This yielded one prediction for feature '{target_feature}' in ticker {ticker}:")
    print(prediction[0])
    print(f"This means we expect the price today to be: {(1 + prediction[0] / 100) * days_back_close}")

    # prediction is over, and the {horizon}-day-prediction-csv folder already has a csv in it, let's change the value!

    return prediction[0]
