# the goal of this file is just the loading the most up-to-date historical CSV file of an array of stocks for further analysis
import pandas as pd
import numpy as np
import requests 
import yfinance as yf
from datetime import datetime

tickers = [
    'AAPL',
    'AMZN',
    'INTC',
    'MSFT',
    'QQQ',
    'SH',
    'SPY',
    'TEAM',
    'XYLD'
]

unsafe_session = requests.session()
unsafe_session.verify = False

def load_csv_files():
    for ticker in tickers:
        load_data(ticker)

        now = datetime.now()
        timestamp = now.strftime("%H:%M %m/%d/%Y")

        print(f"Loading historical data for '{ticker}' as of {timestamp}")

def load_data(tickerSymbols):
    yf.download(tickers=tickerSymbols
                , session=unsafe_session
                ).to_csv(f'./csv/{tickerSymbols}_data.csv')

    # load data into DataFrame
    return pd.read_csv(f'./csv/{tickerSymbols}_data.csv')

def get_last_close_price(ticker):
    # last close price will ALWAYS be the final line of the CSV

    #load data into a DataFrame
    df = pd.read_csv(f'./csv/{ticker}_data.csv')

    return df.loc[len(df)-1]['Close']

print(get_last_close_price('SPY'))

# load_csv_files()
