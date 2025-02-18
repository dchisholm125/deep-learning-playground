import os
from matplotlib.dates import relativedelta
import pandas as pd
import joblib
import pandas_ta as ta
import numpy as np
import requests 
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

def train_single_model(ticker, timestamp, target_feature):
    if(os.path.exists(f"./models/{ticker}-model-{target_feature}-{timestamp}.joblib") and os.path.exists(f"./models/{ticker}-X-{target_feature}-{timestamp}.joblib")):
        print('Model found, DON\'T TRAIN!')
        return

    df = pd.read_csv(f"./modified-csv/{ticker}_shared_{timestamp}.csv")

    model = RandomForestRegressor(random_state=1)

    X_features = []
    
    # remove {feature} from generating X_features so it's only in y_features
    for feature in df.columns:
        if (feature == 'Price' or feature == target_feature):
            continue
        else:
            X_features.append(feature)

    y_features = [target_feature]

    X = df[X_features]

    y = df[y_features]

    print(f'Model training started for \'{target_feature}\'.')

    model.fit(X[:-2], y[:-2])

    joblib.dump(model, f'./models/{ticker}-model-{target_feature}-{timestamp}.joblib')
    joblib.dump(X, f'./models/{ticker}-X-{target_feature}-{timestamp}.joblib')

    print(f'Model training ended for \'{target_feature}\'. Model and X saved to \'/models/\'')

    return model, X
