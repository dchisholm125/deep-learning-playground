import os
from matplotlib.dates import relativedelta
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import yfinance as yf

def generate_model_consumable_csvs(ticker, timestamp, years_back, horizon = 30, back_buffer_num = 50, folder_name = 'modified-csv'):
    download_raw_stock_csv(ticker, timestamp)
    create_csv_copy(ticker, timestamp, folder_name)
    reduce_copied_csv(ticker, timestamp, years_back, back_buffer_num, folder_name)
    calc_csv_technicals(ticker, timestamp, folder_name)
    add_csv_lagged_features(ticker, timestamp, folder_name)
    remove_buffered_rows(ticker, timestamp, back_buffer_num, folder_name)
    create_prediction_csv(ticker, timestamp, folder_name, horizon)

def download_raw_stock_csv(ticker, timestamp):
        """
        Download raw CSV from yfinance by ticker symbol.
        """
        unsafe_session = requests.session()
        unsafe_session.verify = False

        yf.download(tickers=ticker
                    , session=unsafe_session
                    ).to_csv(f'./csv/{ticker}_data_{timestamp}.csv')
        
        print(f'Raw stock data downloaded for {ticker} on {timestamp}.')
        
def create_csv_copy(ticker, timestamp, folder_name = 'modified-csv'):
        """
        Create a copy of an existing file containing raw stock data from yfinance. 
        """
        raw_csv_path = f'./csv/{ticker}_data_{timestamp}.csv'

        if(os.path.exists(raw_csv_path)):
            df = pd.read_csv(raw_csv_path)

            df.to_csv(f'./{folder_name}/{ticker}_shared_{timestamp}.csv', index=False)
        else:
            print(f'CSV doesn\'t exist for: \"{raw_csv_path}\"')

def reduce_copied_csv(ticker, timestamp, years_back, back_buffer_num = 50, folder_name = 'modified-csv'):
        """
        Reduce a CSV's stock data to window for models to consume
        """
        today = datetime.today()
        x_yrs_ago = today - relativedelta(years=years_back)
        x_yrs_string = x_yrs_ago.strftime("%Y-%m-%d")

        copied_csv_path = f'./{folder_name}/{ticker}_shared_{timestamp}.csv'

        if(os.path.exists(copied_csv_path)):
            # load the data into a dataframe
            df = pd.read_csv(copied_csv_path)

            # if not exact, try going back by calculating the index
            try:
                # grab the INDEX of the row containing the string of the date we're looking for
                index = df.loc[df['Price'] == x_yrs_string].index.tolist()[0]
                # go back by buffer amount, which can be dynamic
                index -= back_buffer_num
                df = df.loc[index:]

                print(f"Exact date found, CSV modified to only include data back as far as {x_yrs_string}")
            except:
                # estimate the index and ADD buffer for backwards .loc[] lookup
                indices_back = years_back * ((52 * 5) - 8) + back_buffer_num
                df = df.loc[-indices_back:]

                print(f"No exact date found, estimating date instead.\n CSV modified to include {indices_back} rows")

            # modify the CSV file AGAIN!
            df.to_csv(copied_csv_path, index=False)

def calc_csv_technicals(ticker, timestamp, folder_name = 'modified-csv'):
    """
    Create technical indicators on copied CSV file.
    """
    copied_csv_path = f'./{folder_name}/{ticker}_shared_{timestamp}.csv'

    df = pd.read_csv(copied_csv_path)
    df = df[2:]

    # convert all CSV columns to a numerics
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    # Calculate returns
    df['Return'] = (df['Close'].pct_change()) * 100 # percentage change 

    # Compute technical indicators
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["RSI"] = ta.rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = ta.macd(df["Close"]).T.values
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"])
    df.dropna()

    # modify the CSV again
    df.to_csv(copied_csv_path, index=False)

### Create 30 days of lagged features for model to consume
def add_csv_lagged_features(ticker, timestamp, folder_name = 'modified-csv'):
    """
    Add lagged features up to `horizon` days in the past, for every `ticker` in `tickers
    """
    copied_csv_path = f'./{folder_name}/{ticker}_shared_{timestamp}.csv'

    # load the data into a dataframe
    df = pd.read_csv(copied_csv_path)

    feature_list = df.columns[1:] # exclude the 'Price' column (which is really the 'Date' col anyway, but yfinance is weird)

    # only create 30 lagged days worth of features for model to build insights off of
    for feature in feature_list:
        for i in range(1, 30 + 1):
            new_feat = f"{feature}_lag{i}"
            df[new_feat] = df[feature].shift(i)

    df.to_csv(copied_csv_path, index=False)

def remove_buffered_rows(ticker, timestamp, back_buffer_num = 50, folder_name = 'modified-csv'):
    """
    Remove the first 50 or `back_buffer_num` rows from modified CSV file.
    """
    copied_csv_path = f'./{folder_name}/{ticker}_shared_{timestamp}.csv'

    df = pd.read_csv(copied_csv_path)

    df[back_buffer_num:].to_csv(copied_csv_path, index=False)

def create_prediction_csv(ticker, timestamp, folder_name = 'modified-csv', horizon = 30):
    """
    Create a new prediction CSV for models to add predictions to.
    """
    copied_csv_path = f'./{folder_name}/{ticker}_shared_{timestamp}.csv'
    prediction_folder = f'{horizon}-day-prediction-csv'
    prediction_csv_path = f'./{prediction_folder}/{ticker}_{horizon}_day_prediction_{timestamp}.csv'

    df = pd.read_csv(copied_csv_path)
    df = df[-1:]

    # also create a prediction CSV at this time, from the final row
    # 1. check if folder exists, if not, create it:
    if not os.path.exists(f"./{prediction_folder}"):
        os.makedirs(f"./{prediction_folder}")

    df.loc[-1:, 'Price'] = 0

    df.to_csv(prediction_csv_path, index=False)

    print(f'Prediction CSV generated: \'{prediction_csv_path}\'') 

def copy_csv_last_row(file_path):
     """
     Copies the last row of a CSV for further manipulation via a DataFrame.
     """
     df = pd.read_csv(file_path)

     print('BEFORE!')
     print(df)

     df.loc[len(df)] = df.loc[len(df)-1]
     df.loc[len(df)-1, 'Price'] += 1

     print('AFTER!')
     print(df)

     df.to_csv(file_path, index=False)

def copy_and_move_last_row(copy_file_path, dest_file_path):
    df_copy = pd.read_csv(copy_file_path)
    df_dest = pd.read_csv(dest_file_path)

    df_dest.loc[len(df_dest)] = df_copy.loc[len(df_copy)-1]

    df_dest.to_csv(dest_file_path, index=False)

def move_back_lagged_features(ticker, horizon, timestamp):
    """
    Moves all lagged features from {feature}_lag1..29 >>> {feature}_lag2..30 in prediction CSV.
    """
    base_features = ['Close','High','Low','Open','Volume','Return','SMA_50','RSI','MACD','MACD_Signal','MACD_Hist','ATR']
    
    prediction_csv_file_path = f"./{horizon}-day-prediction-csv/{ticker}_{horizon}_day_prediction_{timestamp}.csv"

    # first, move all "{feature}_lag1..29" to "{feature}_lag2..30"
    df_orig = pd.read_csv(prediction_csv_file_path)

    # copy so we don't have weird renaming issues
    df = df_orig[len(df_orig)-1:].copy()

    # for 
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

    df.loc[-1:, 'Price'] += 1

    df = pd.concat([df_orig, df])[1:]

    df.to_csv(prediction_csv_file_path, index = False)
