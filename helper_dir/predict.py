import joblib
import pandas as pd
from helper_dir import model_helpers as mh

def make_prediction(ticker, timestamp, target_feature):
    model = joblib.load(f"./models/{ticker}-model-{target_feature}-{timestamp}.joblib")
    X = joblib.load(f"./models/{ticker}-X-{target_feature}-{timestamp}.joblib")

    prediction = model.predict(X[-1:])

    print('Model fit up to the SECOND to last row.')
    print(f"Model was asked to make a prediction from the LAST row of information. This yielded one prediction for feature {target_feature}:")
    print(prediction[0])

    # prediction is over, and the {horizon}-day-prediction-csv folder already has a csv in it, let's change the value!

    return prediction[0]

def add_predictions_to_csv(ticker, timestamp, horizon, features):
    df = pd.read_csv(f"./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv")

    for feature in features:
        mh.train_single_model(ticker, timestamp, feature)
        prediction = make_prediction(ticker, timestamp, feature)

        # edit lines in-place
        df.loc[-1:, feature] = prediction

    df.to_csv(f"./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv", index=False)
