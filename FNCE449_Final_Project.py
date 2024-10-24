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


# IMPLEMENTING A PROFIT MODEL FOR EACH STRATEGY
# strategy 1
stockA_Trades['profit_strategy1'] = stockA_Trades['price'].shift(-1) - stockA_Trades['price']
stockB_Trades['profit_strategy1'] = stockB_Trades['price'].shift(-1) - stockB_Trades['price']
stockC_Trades['profit_strategy1'] = stockC_Trades['price'].shift(-1) - stockC_Trades['price']

#strategy 2
stockA_Trades['profit_strategy2'] = stockA_Trades['price'].shift(-1) - stockA_Trades['price']
stockB_Trades['profit_strategy2'] = stockB_Trades['price'].shift(-1) - stockB_Trades['price']
stockC_Trades['profit_strategy2'] = stockC_Trades['price'].shift(-1) - stockC_Trades['price']

# label the strategies (strategy 1 = 1, strategy 2 = 0)
stockA_Trades['strategy_label'] = np.where(stockA_Trades['strategy1'] == 1, 1, 0)
stockB_Trades['strategy_label'] = np.where(stockB_Trades['strategy1'] == 1, 1, 0)
stockC_Trades['strategy_label'] = np.where(stockC_Trades['strategy1'] == 1, 1, 0)


# PROPENSITY MATCHING
x_A = stockA_Trades[['price', 'volume', 'momentum']] # using the features I want to base propensity on
y_A = stockA_Trades['strategy_label'] # set y to be the strategy label which indicates if it is strategy 1 or 2

x_B = stockB_Trades[['price', 'volume', 'momentum']]
y_B = stockB_Trades['strategy_label']

x_C = stockC_Trades[['price', 'volume', 'momentum']]
y_C = stockC_Trades['strategy_label']

logistic_model_A = LogisticRegression()
logistic_model_A.fit(x_A, y_A)

logistic_model_B = LogisticRegression()
logistic_model_B.fit(x_B, y_B)

logistic_model_C = LogisticRegression()
logistic_model_C.fit(x_C, y_C)

# calculate propensity score by predicting class probabilities
stockA_Trades['propensity_score'] = logistic_model_A.predict_proba(x_A)[:, 1]
stockB_Trades['propensity_score'] = logistic_model_B.predict_proba(x_B)[:, 1]
stockC_Trades['propensity_score'] = logistic_model_C.predict_proba(x_C)[:, 1]

# use nearest neighbors to match trades
treated_A = stockA_Trades[stockA_Trades['strategy_label'] == 1] # the "treated" observations are those that used strategy 1
treated_B = stockB_Trades[stockB_Trades['strategy_label'] == 1]
treated_C = stockC_Trades[stockC_Trades['strategy_label'] == 1]

control_A = stockA_Trades[stockA_Trades['strategy_label'] == 0] # the "control" observations are those that used strategy 2
control_B = stockB_Trades[stockB_Trades['strategy_label'] == 0]
control_C = stockC_Trades[stockC_Trades['strategy_label'] == 0]

nn_A = NearestNeighbors(n_neighbors = 1) # apply the Nearest Neighbors model
nn_A.fit(control_A[['propensity_score']]) # fit the Nearest Neighbors model on the control dataset

nn_B = NearestNeighbors(n_neighbors = 1)
nn_B.fit(control_B[['propensity_score']])

nn_C = NearestNeighbors(n_neighbors = 1)
nn_C.fit(control_C[['propensity_score']])

# find the closest control match for each treated observation
distances_A, indices_A = nn_A.kneighbors(treated_A[['propensity_score']])
distances_B, indices_B = nn_B.kneighbors(treated_B[['propensity_score']])
distances_C, indices_C = nn_C.kneighbors(treated_C[['propensity_score']])

# get the matched control data
matched_control_A = control_A.iloc[indices_A.flatten()]
matched_control_B = control_B.iloc[indices_B.flatten()]
matched_control_C = control_C.iloc[indices_C.flatten()]

# concatenate the treated and control data for matched pairs
matched_pairs_A = pd.concat([treated_A.reset_index(drop = True), matched_control_A.reset_index(drop = True)], axis = 1, keys = ['Treated', 'Control'])
matched_pairs_B = pd.concat([treated_B.reset_index(drop = True), matched_control_B.reset_index(drop = True)], axis = 1, keys = ['Treated', 'Control'])
matched_pairs_C = pd.concat([treated_C.reset_index(drop = True), matched_control_C.reset_index(drop = True)], axis = 1, keys = ['Treated', 'Control'])
