# Saad Abdullah
# UCID: 30142511
# November 4, 2024
# Python Version 3.11.9
# FNCE 449 Final Project
# Description: This project looks to address which strategy (between the 50-day moving average and simple daily price momentum) is best to use in the 
# technology equities market. It creates long-short portfolios as well as works with Sharpe Ratios to try and answer this question.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

my_api_key="db-RqnJGkKgqELRSkKeK8XAsHktsknGe" # API key for Databento
dataset="XNAS.ITCH" # this is the identifier for NASDAQ
list_symbols = ['MSFT','EA'] # list of stocks
list_csvs = ['msft.csv','ea.csv'] # the 2 csvs the respective dataframes are saved into
schema='ohlcv-1d' # specify the schema
start='2022-09-01' # dates are from September 1, 2022 - September 1, 2024
end='2024-09-01'

# This code is commented out because it was only used during the first execution to save the 3 dataframes to seperate csvs for future executions to avoid charging
# Databento again and again
# for symbol, csv in zip(list_symbols,list_csvs):
#     df = get_range(my_api_key, dataset, symbol, schema, start, end) # call user-defined function
#     df['ts_event']=pd.to_datetime(df['ts_event'])
#     df['close'] = df['close'].map(float) # change prices to a float
#     df.to_csv(csv) # save the dataframe to the specified csv

# for later executions, I read from the csvs on GitHub and converted them into seperate dataframes
url1 = 'https://raw.githubusercontent.com/Saad2814/FNCE-449/refs/heads/main/Data/msft.csv'
url2 = 'https://raw.githubusercontent.com/Saad2814/FNCE-449/refs/heads/main/Data/ea.csv'
stockA_prices = pd.read_csv(url1) # stock A dataframe
stockB_prices = pd.read_csv(url2) # stock B dataframe 

# convert all dates/times to date time form
stockA_prices['ts_event'] = pd.to_datetime(stockA_prices['ts_event'])
stockB_prices['ts_event'] = pd.to_datetime(stockB_prices['ts_event'])

# convert all close prices to floats
stockA_prices['close'] = stockA_prices['close'].map(float)
stockB_prices['close'] = stockB_prices['close'].map(float)

# only keep needed columns
stockA_prices = stockA_prices.loc[:, ['ts_event', 'symbol', 'close']]
stockB_prices = stockB_prices.loc[:, ['ts_event', 'symbol', 'close']]

stocks = stockA_prices.merge(stockB_prices, on='ts_event', suffixes=('_A','_B'), how='inner') # merge on ts_event

# DEFINING THE TWO TRADING STRATEGIES
# note that both stratgies are similar, but the first strategy is over a much longer time period
stocks['50_Period_MA'] = stocks['close_A'].rolling(window=50).mean() # first strategy is the 50-day moving average based on stock A's prices
stocks['price_momentum'] = stocks['close_A'].diff() # second strategy is a price difference strategy based on momentum (using stock A's prices)

# Drop NaNs. I do this instead of back filling to have more accuracy rather than skewing the data.
stocks = stocks.dropna().reset_index(drop=True)

# the following lines were used for testing when I would print out the dataframes
# pd.set_option('display.max_columns', None) # show all columns when printing to terminal
# pd.set_option('display.max_rows', None) # show all rows
# pd.set_option('expand_frame_repr', False) # show all in one line in the terminal rather than splitting into multiple lines

# this function implements strategy1 (50-day moving average)
def implement_strategy1(stocks_data):
    stocks_copy = stocks_data.copy(deep = True) # create a deepcopy of stocks

    # initialize these columns to 0 for portfolio tracking
    stocks_copy['stock_position'] = 0  # 1 for long stock A and short stock B, -1 for short stock A and long stock B
    stocks_copy['cash_flows'] = 0
    stocks_copy['mark_to_market'] = 0
    stocks_copy['cum_cash'] = 0
    stocks_copy['total_trades'] = 0
    
    # define the long and short conditions
    long_condition = stocks_copy['close_A'] > stocks_copy['50_Period_MA']
    short_condition = stocks_copy['close_A'] <= stocks_copy['50_Period_MA']

    # calculate the stock positions based on the above conditions
    stocks_copy['stock_position'] = np.where(long_condition, 1, stocks_copy['stock_position'])
    stocks_copy['stock_position'] = np.where(short_condition, -1, stocks_copy['stock_position'])
    stocks_copy.loc[stocks_copy.index[-1], 'stock_position'] = 0  # final position should be 0 as it is closed out

    # mark rows where the the stock_position changed by comparing that row's stock position to the prior row's stock position
    stocks_copy['position_change'] = stocks_copy['stock_position'] != stocks_copy['stock_position'].shift(1)

    # count total trades
    stocks_copy.loc[0, 'total_trades'] = 1 # update the trade count for the first row
    # if the position changes in any other rows, also update the trade count
    stocks_copy.iloc[1:, stocks_copy.columns.get_loc('total_trades')] = stocks_copy['position_change'].iloc[1:].astype(int) * 2
    stocks_copy['total_trades'] = stocks_copy['total_trades'].cumsum() # use cumulative sum to carry over the total_trades row by row
    # update the trade count in the last row
    stocks_copy.loc[stocks_copy.index[-1], 'total_trades'] = stocks_copy.loc[stocks_copy.index[-2], 'total_trades'] + 1

    # calculate cash flows by closing out the prior position and taking the new position --> there are only cash flows when trades occur
    stocks_copy.loc[stocks_copy['position_change'], 'cash_flows'] = stocks_copy['close_A'] * stocks_copy['stock_position'] * -1 * 2 + stocks_copy['close_B'] * stocks_copy['stock_position'] * 2
    # in the first row, there is only 1 trade so account for this
    stocks_copy.loc[stocks_copy.index[0], 'cash_flows'] = stocks_copy.loc[stocks_copy.index[0], 'close_A'] * stocks_copy.loc[stocks_copy.index[0], 'stock_position'] * -1 + stocks_copy.loc[stocks_copy.index[0], 'close_B'] * stocks_copy.loc[stocks_copy.index[0], 'stock_position']
    stocks_copy['mark_to_market'] = stocks_copy['close_A'] * stocks_copy['stock_position'] + stocks_copy['close_B'] * stocks_copy['stock_position'] * -1 # calculate mark-to-market
    # close out position in the last row by updating cash flows
    stocks_copy.loc[stocks_copy.index[-1], 'cash_flows'] = stocks_copy['close_A'].iloc[-1] * stocks_copy['stock_position'].iloc[-2] + stocks_copy['close_B'].iloc[-1] * stocks_copy['stock_position'].iloc[-2] * -1
    stocks_copy['cum_cash'] = stocks_copy['cash_flows'].cumsum() # calculate cumulative cash flows
    # update portfolio value by adding together cum_cash and mark_to_market
    stocks_copy['portfolio_value'] = stocks_copy['cum_cash'] + stocks_copy['mark_to_market']

    return stocks_copy


# this function implements strategy2 (simple price momentum strategy) for each stock
def implement_strategy2(stocks_data): 
    stocks_copy = stocks_data.copy(deep = True) # create a deepcopy of stocks

    # initialize these columns to 0 for portfolio tracking
    stocks_copy['stock_position'] = 0  # 1 for long stock A and short stock B, -1 for short stock A and long stock B
    stocks_copy['cash_flows'] = 0
    stocks_copy['mark_to_market'] = 0
    stocks_copy['cum_cash'] = 0
    stocks_copy['total_trades'] = 0
    
    # define the long and short conditions
    long_condition = stocks_copy['price_momentum'] > 0
    short_condition = stocks_copy['price_momentum'] <= 0

    # calculate the stock positions based on the above conditions
    stocks_copy['stock_position'] = np.where(long_condition, 1, stocks_copy['stock_position'])
    stocks_copy['stock_position'] = np.where(short_condition, -1, stocks_copy['stock_position'])
    stocks_copy.loc[stocks_copy.index[-1], 'stock_position'] = 0  # final position should be 0 as it is closed out

    # mark rows where the the stock_position changed by comparing that row's stock position to the prior row's stock position
    stocks_copy['position_change'] = stocks_copy['stock_position'] != stocks_copy['stock_position'].shift(1)

    # count total trades
    stocks_copy.loc[0, 'total_trades'] = 1 # update the trade count for the first row
    # if the position changes in any other rows, also update the trade count
    stocks_copy.iloc[1:, stocks_copy.columns.get_loc('total_trades')] = stocks_copy['position_change'].iloc[1:].astype(int) * 2
    stocks_copy['total_trades'] = stocks_copy['total_trades'].cumsum() # use cumulative sum to carry over the total_trades row by row
    # update the trade count in the last row
    stocks_copy.loc[stocks_copy.index[-1], 'total_trades'] = stocks_copy.loc[stocks_copy.index[-2], 'total_trades'] + 1

    # calculate cash flows by closing out the prior position and taking the new position --> there are only cash flows when trades occur
    stocks_copy.loc[stocks_copy['position_change'], 'cash_flows'] = stocks_copy['close_A'] * stocks_copy['stock_position'] * -1 * 2 + stocks_copy['close_B'] * stocks_copy['stock_position'] * 2
    # in the first row, there is only 1 trade so account for this
    stocks_copy.loc[stocks_copy.index[0], 'cash_flows'] = stocks_copy.loc[stocks_copy.index[0], 'close_A'] * stocks_copy.loc[stocks_copy.index[0], 'stock_position'] * -1 + stocks_copy.loc[stocks_copy.index[0], 'close_B'] * stocks_copy.loc[stocks_copy.index[0], 'stock_position']
    stocks_copy['mark_to_market'] = stocks_copy['close_A'] * stocks_copy['stock_position'] + stocks_copy['close_B'] * stocks_copy['stock_position'] * -1 # calculate mark-to-market
    # close out position in the last row by updating cash flows
    stocks_copy.loc[stocks_copy.index[-1], 'cash_flows'] = stocks_copy['close_A'].iloc[-1] * stocks_copy['stock_position'].iloc[-2] + stocks_copy['close_B'].iloc[-1] * stocks_copy['stock_position'].iloc[-2] * -1
    stocks_copy['cum_cash'] = stocks_copy['cash_flows'].cumsum() # calculate cumulative cash flows
    # update portfolio value by adding together cum_cash and mark_to_market
    stocks_copy['portfolio_value'] = stocks_copy['cum_cash'] + stocks_copy['mark_to_market']

    return stocks_copy

# Implement strategies
stocks_strategy1 = implement_strategy1(stocks)
stocks_strategy2 = implement_strategy2(stocks)

# this chart is to show the price of each stock over time
plt.figure(figsize=(10, 6))
# prices of stock A
plt.plot(stocks['ts_event'], stocks['close_A'], label='Price - ' + str(stocks.iloc[0]['symbol_A']), color='blue')
# prices of stock B
plt.plot(stocks['ts_event'], stocks['close_B'], label='Price - ' + str(stocks.iloc[0]['symbol_B']), color='green')
plt.title('Closing Prices of Stocks Over Time - ' + str(stocks.iloc[0]['symbol_A']) + ' & ' + str(stocks.iloc[0]['symbol_B']), fontsize=14)
# formatting
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Prices ($)', fontsize=14)
plt.legend()
plt.show()

# this chart is to show how the 50 Day Moving Average smooths out the price fluctuations
plt.figure(figsize=(10, 6))
# Plot 50 Day Moving Average for stock A
plt.plot(stocks_strategy1['ts_event'], stocks_strategy1['50_Period_MA'], color='blue')
# Plot Daily Price Momentum for stock A
plt.plot(stocks_strategy1['ts_event'], stocks_strategy1['price_momentum'], color='green')
# formatting
plt.title('50-Day Moving Average & Daily Price Momentum - ' + str(stocks_strategy1.iloc[0]['symbol_A']), fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Value ($)', fontsize=14)
plt.legend()
plt.show()

# this chart is to show the long-short portfolio value over time for each strategy
plt.figure(figsize=(10,6))
# strategy1 portfolio value over time
plt.plot(stocks_strategy1['ts_event'], stocks_strategy1['portfolio_value'], label='Strategy 1 - ' + str(stocks_strategy1.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy1.iloc[0]['symbol_B']), color='blue')
# strategy2 portfolio value over time
plt.plot(stocks_strategy2['ts_event'], stocks_strategy2['portfolio_value'], label='Strategy 2 - ' + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B']), color='green')
plt.title('Long-Short Portfolio Value Over Time - ' + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B']), fontsize=14)
# formatting
plt.xlabel('Date', fontsize=14)
plt.ylabel('Portfolio Value ($)', fontsize=14)
plt.legend()
plt.show()

# this chart is to show the total number of trades over time for each strategy for each stock
plt.figure(figsize=(10, 6))
# strategy1 number of trades
plt.plot(stocks_strategy1['ts_event'], stocks_strategy1['total_trades'], label='Strategy 1 - ' + str(stocks_strategy1.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy1.iloc[0]['symbol_B']), color='blue')
# strategy2 number of trades
plt.plot(stocks_strategy2['ts_event'], stocks_strategy2['total_trades'], label='Strategy 2 - ' + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B']), color='green')
plt.title('Number of Trades Over Time - ' + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B']), fontsize=14)
# formatting
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Trades', fontsize=14)
plt.legend()
plt.show()

# read in the data for the daily risk free rate --> based on 10-Year US Treasury Note
url3 = 'https://raw.githubusercontent.com/Saad2814/FNCE-449/refs/heads/main/Data/risk_free_rates.csv'
rf_rate = pd.read_csv(url3)
rf_rate['ts_event'] = pd.to_datetime(rf_rate['ts_event']) # change ts_event to date time format
rf_rate['rate'] = rf_rate['rate'].map(float) # convert daily risk free rates to a float

# this function is used to calculate annualized Sharpe Ratios
def calculate_sharpe_ratio(stocks_data):
    # calculate portfolio return --> abs value is used to deal with the case where we go from a negative portfolio value to a positive
    # ^ the pct_change() function doesn't deal correctly with this situation
    stocks_data['portfolio_growth'] = stocks_data['portfolio_value'].diff() / abs(stocks_data['portfolio_value'].shift(1))
    stocks_data = stocks_data.merge(rf_rate, on='ts_event', how='inner') # merge the daily risk free rates with the stocks data
    stocks_data['excess_returns'] = stocks_data['portfolio_growth'] - stocks_data['rate'] # calculate excess returns
    excess_returns_average = stocks_data['excess_returns'].mean() # take the average of the excess returns
    excess_returns_std = stocks_data['excess_returns'].std() # take the standard deviation of the excess returns
    sharpe_ratio = excess_returns_average / excess_returns_std # calculate Sharpe Ratio
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252) # annualize the Sharpe Ratio since we used daily observations
    return annualized_sharpe_ratio

# remove time from ts_event in stocks_strategy 1 and 2 and change to datetime form to match ts_event from rf_rate
stocks_strategy1['ts_event'] = stocks_strategy1['ts_event'].dt.date
stocks_strategy2['ts_event'] = stocks_strategy2['ts_event'].dt.date
stocks_strategy1['ts_event'] = pd.to_datetime(stocks_strategy1['ts_event'])
stocks_strategy2['ts_event'] = pd.to_datetime(stocks_strategy2['ts_event'])

# calculate annualized Sharpe Ratios for each strategy
sharpe_ratio_strategy1 = calculate_sharpe_ratio(stocks_strategy1)
sharpe_ratio_strategy2 = calculate_sharpe_ratio(stocks_strategy2)

# plot the Sharpe Ratios of both strategies in a bar chart
sharpe_ratios = [sharpe_ratio_strategy1, sharpe_ratio_strategy2]
strategies = ['Strategy 1 - ' + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B']), 'Strategy 2 - ' + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B'])]
plt.figure(figsize=(10, 6))
# each bar will represent the annualized Sharpe Ratio for a strategy
plt.bar(strategies, sharpe_ratios, color=['blue', 'green'])
# formatting
plt.xlabel("Trading Strategies", fontsize=14)
plt.ylabel("Annualized Sharpe Ratio", fontsize=14)
plt.title("Annualized Sharpe Ratios of Trading Strategies - " + str(stocks_strategy2.iloc[0]['symbol_A']) + ' & ' + str(stocks_strategy2.iloc[0]['symbol_B']), fontsize=16)
plt.show()
