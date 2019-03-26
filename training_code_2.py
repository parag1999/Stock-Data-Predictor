import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Initialize the plot figure
# Initialize the short and long windows
short_window = 40
long_window = 100
import quandl 
def Get_Stock_data(symbol):
    quandl.ApiConfig.api_key = "aVf72bQcXycLqzBV3zCf"
    aapl = quandl.get("WIKI/{}".format(symbol), start_date="2016-03-01", end_date="2017-03-01")
    if aapl.empty:
        return 0
    aapl.to_csv('database_2/{}.csv'.format(symbol))
    df = pd.read_csv('database_2/{}.csv'.format(symbol), header=0, index_col='Date', parse_dates=True)
    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=aapl.index)
    signals['signal'] = 0.0
    
    # Create short simple moving average over the short window
    signals['short_mavg'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    
    # Create long simple moving average over the long window
    signals['long_mavg'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)   
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    
    fig = plt.figure(figsize=(8,4))
    
    # Add a subplot and label for y-axis
    ax1 = fig.add_subplot(111,  ylabel='Price')
    
    # Plot the closing price
    aapl['Close'].plot(ax=ax1, color='r', lw=2.)
    
    # Plot the short and long moving averages
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
    
    # Plot the buy signals
    ax1.plot(signals.loc[signals.positions == 1.0].index, 
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='y')
             
    # Plot the sell signals
    ax1.plot(signals.loc[signals.positions == -1.0].index, 
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k')
             
    plt.savefig('images/{}.png'.format(symbol))
    # Show the plot
    #plt.show()# -*- coding: utf-8 -*-
    
if __name__=="__main__":
    symbols = [ 
            'TMK', 'TSS', 'TSCO', 'RIG', 
            'TRIP', 'FOXA', 'TSN', 'TYC', 
            'UA', 'UNP', 'UNH', 'UPS', 
            'URI', 'UTX', 'UHS', 'UNM', 
            'URBN', 'VFC', 'VLO', 'VAR', 
            'VTR', 'VRSN', 'VZ', 'VRTX', 
            'VIAB', 'V', 'VNO', 'VMC', 
            'WMT', 'WBA', 'DIS', 'WM', 
            'WAT', 'ANTM', 'WFC', 'WDC', 
            'WU', 'WY', 'WHR', 'WFM', 
            'WMB', 'WEC', 'WYN', 'WYNN', 
            'XEL', 'XRX', 'XLNX', 'XL', 
            'XYL', 'YHOO', 'YUM', 'ZBH', 
            'ZION', 'ZTS'
]

for symbol in symbols:    
    Get_Stock_data(symbol)

