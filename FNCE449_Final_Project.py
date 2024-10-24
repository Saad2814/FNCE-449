# Saad Abdullah
# UCID: 30142511
# October 23, 2024
# Python Version 3.11.9
# FNCE 449 Final Project
# Description: 

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# GETTING THE DATA
# user-defined function to get the data from databento
def get_range(api_key,dataset,symbols,schema,start,end,limit=None):
    
    import requests
    from io import StringIO
    from requests.auth import HTTPBasicAuth
    
    url = 'https://hist.databento.com/v0/timeseries.get_range' # link to databento
    auth = HTTPBasicAuth(api_key, '')

    payload = {
        'dataset': dataset,
        'symbols': symbols,
        'schema': schema,
        'start': start,
        'end': end,
        'encoding': 'csv',
        'pretty_px': 'true',
        'pretty_ts': 'true',
        'map_symbols': 'true',
        'limit':limit
    }

    response = requests.post(url, auth=auth, data=payload)
    
    csv_data = StringIO(response.content.decode('utf-8')) # saving it as csv type of data
    df=pd.read_csv(csv_data) # reading the csv data
    
    return df # output the dataframe

my_api_key="db-RqnJGkKgqELRSkKeK8XAsHktsknGe" # this is my personal API key for Databento
dataset="XNAS.ITCH" # this is the identifier for NASDAQ
list_symbols = ['NVDA', 'AAPL', 'MSFT'] # the 3 stocks I am using are Nvidia, Apple, and Microsoft
list_csvs = ['nvda.csv', 'aapl.csv', 'msft.csv'] # the 3 csvs the respective dataframes are saved into
schema='tbbo' # specify the schema
start='2024-08-25' # dates are from September 1, 2022 - September 1, 2024
end='2024-09-01'

for symbol, csv in zip(list_symbols,list_csvs):
    print(symbol)
    print(csv)
    print("\n")
    df = get_range(my_api_key,dataset,symbol,schema,start,end) # call user-defined function
    df['price'] = df['price'].map(float) # change prices to a float
    df.to_csv(csv) # during the first execution, I saved each of the 3 dataframes to a seperate csv to avoid charging DataBento again and again

# for later executions, I read from the csvs and converted them into seperate dataframes
stockA_Trades = pd.read_csv("nvds.csv") # NVDA dataframe
stockB_Trades = pd.read_csv("aapl.csv") # AAPL dataframe
stockC_Trades = pd.read_csv("msft.csv") # MSFT dataframe


# DEFINING THE TWO TRADING STRATEGIES
stockA_Trades['50_Period_MA'] = stockA_Trades['price'].rolling(window=50).mean() # first strategy is the 50-period moving average
stockB_Trades['50_Period_MA'] = stockB_Trades['price'].rolling(window=50).mean()
stockC_Trades['50_Period_MA'] = stockC_Trades['price'].rolling(window=50).mean()

stockA_Trades['price_momentum'] = stockA_Trades['price'].diff() # second strategy is a price difference strategy based on momentum
stockB_Trades['price_momentum'] = stockB_Trades['price'].diff()
stockC_Trades['price_momentum'] = stockC_Trades['price'].diff()

# note that both stratgies are similar, but the first strategy is over a much longer time period

# for strategy 1, buy when the price is > the 50-period moving average and sell otherwise (Buy = 1, Sell = 0)
stockA_Trades['strategy1'] = np.where(stockA_Trades['price'] > stockA_Trades['50_Period_MA'], 1, 0)
stockB_Trades['strategy1'] = np.where(stockB_Trades['price'] > stockB_Trades['50_Period_MA'], 1, 0)
stockC_Trades['strategy1'] = np.where(stockC_Trades['price'] > stockC_Trades['50_Period_MA'], 1, 0)

# for strategy 2, buy when the momentum is > 0 and sell otherwise (Buy = 1, Sell = 0)
stockA_Trades['strategy2'] = np.where(stockA_Trades['price_momentum'] > 0, 1, 0)
stockB_Trades['strategy2'] = np.where(stockB_Trades['price_momentum'] > 0, 1, 0)
stockC_Trades['strategy2'] = np.where(stockC_Trades['price_momentum'] > 0, 1, 0)
