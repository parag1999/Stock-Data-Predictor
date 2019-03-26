#import packages
import pandas as pd
import numpy as np
#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

scaler = MinMaxScaler(feature_range=(0, 1))
#to plot within notebook
import matplotlib.pyplot as plt

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,5

#for normalizing data
def Get_Stock_Future(symbol):
#read the file
    df = pd.read_csv('database_2/{}.csv'.format(symbol))
    
    
    #setting index as date
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']
    
    #plot
    #plt.figure(figsize=(8,4))
    plt.plot(df['Close'], label='Close Price history')
    
    
    
    #creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
    
    #setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    
    #creating train and test sets
    dataset = new_data.values
    length = int(len(dataset)/2)
    train = dataset[0:length,:]
    valid = dataset[length:,:]
    
    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)
    
    #predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    train = new_data[:length]
    valid = new_data[length:]
    valid['Predictions'] = closing_price
    
    ax = train['Close'].plot()
    valid['Predictions'].plot(ax=ax,title = "{}".format(symbol))
    plt.show()

if __name__=="__main__":
    symbols = ['ABT', 'ABBV', 'ACN', 'ACE',
           'ADBE', 'ADT', 'AAP', 'AES',
           'AET', 'AFL', 'AMG', 'A',
           'GAS', 'APD', 'ARG', 'AKAM',
           'AA', 'AGN', 'ALXN', 'ALLE',
           'ADS', 'ALL', 'ALTR', 'MO',
           'AMZN', 'AEE', 'AAL', 'AEP',
           'AXP', 'AIG', 'AMT', 'AMP',
           'ABC', 'AME', 'AMGN', 'APH',
           'APC', 'ADI', 'AON', 'APA',
           'AIV', 'AMAT', 'ADM', 'AIZ',
           'T', 'ADSK', 'ADP', 'AN',
           'AZO', 'AVGO', 'AVB', 'AVY',
           'BHI', 'BLL', 'BAC', 'BK',
           'BCR', 'BXLT', 'BAX', 'BBT',
           'BDX', 'BBBY', 'BRK-B',
           'BBY', 'BLX', 'HRB', 'BA',
           'BWA', 'BXP', 'BSK', 'BMY',
           'BRCM', 'BF-B', 'CHRW',
           'CA', 'CVC', 'COG', 'CAM',
           'CPB', 'COF', 'CAH', 'HSIC',
           'KMX', 'CCL', 'CAT', 'CBG',
           'CBS', 'CELG', 'CNP', 'CTL',
           'CERN', 'CF', 'SCHW', 'CHK',
           'CVX', 'CMG', 'CB', 'CI',
           'XEC', 'CINF', 'CTAS', 'CSCO',
           'C', 'CTXS', 'CLX', 'CME', 'CMS',
           'COH', 'KO', 'CCE', 'CTSH', 'CL',
           'CMCSA', 'CMA', 'CSC', 'CAG',
           'COP', 'CNX', 'ED', 'STZ', 'GLW',
           'COST', 'CCI', 'CSX', 'CMI',
           'CVS', 'DHI', 'DHR', 'DRI',
           'DVA', 'DE', 'DLPH', 'DAL',
           'XRAY', 'DVN', 'DO', 'DTV',
           'DFS', 'DISCA', 'DISCK', 'DG',
           'DLTR', 'D', 'DOV', 'DOW',
           'DPS', 'DTE', 'DD', 'DUK',
           'DNB', 'ETFC', 'EMN', 'ETN',
           'EBAY', 'ECL', 'EIX', 'EW',
           'EA', 'EMC', 'EMR', 'ENDP',
           'ESV', 'ETR', 'EOG', 'EQT',
           'EFX', 'EQIX', 'EQR', 'ESS',
           'EL', 'ES', 'EXC', 'EXPE',
           'EXPD', 'ESRX', 'XOM', 'FFIV',
           'FB', 'FAST', 'FDX', 'FIS',
           'FITB', 'FSLR', 'FE', 'FSIV',
           'FLIR', 'FLS', 'FLR', 'FMC',
           'FTI', 'F', 'FOSL', 'BEN',
           'FCX', 'FTR', 'GME', 'GPS',
           'GRMN', 'GD', 'GE', 'GGP',
           'GIS', 'GM', 'GPC', 'GNW',
           'GILD', 'GS', 'GT', 'GOOGL',
           'GOOG', 'GWW', 'HAL', 'HBI',
           'HOG', 'HAR', 'HRS', 'HIG',
           'HAS', 'HCA', 'HCP', 'HCN',
           'HP', 'HES', 'HPQ', 'HD',
           'HON', 'HRL', 'HSP', 'HST',
           'HCBK', 'HUM', 'HBAN', 'ITW',
           'IR', 'INTC', 'ICE', 'IBM',
           'IP', 'IPG', 'IFF', 'INTU',
           'ISRG', 'IVZ', 'IRM', 'JEC',
           'JBHT', 'JNJ', 'JCI', 'JOY',
           'JPM', 'JNPR', 'KSU', 'K',
           'KEY', 'GMCR', 'KMB', 'KIM',
           'KMI', 'KLAC', 'KSS', 'KRFT',
           'KR', 'LB', 'LLL', 'LH',
           'LRCX', 'LM', 'LEG', 'LEN',
           'LVLT', 'LUK', 'LLY', 'LNC',
           'LLTC', 'LMT', 'L', 'LOW',
           'LYB', 'MTB', 'MAC', 'M',
           'MNK', 'MRO', 'MPC', 'MAR',
           'MMC', 'MLM', 'MAS', 'MA',
           'MAT', 'MKC', 'MCD', 'MHFI',
           'MCK', 'MJN', 'MMV', 'MDT',
           'MRK', 'MET', 'KORS', 'MCHP',
           'MU', 'MSFT', 'MHK', 'TAP',
           'MDLZ', 'MON', 'MNST', 'MCO',
           'MS', 'MOS', 'MSI', 'MUR',
           'MYL', 'NDAQ', 'NOV', 'NAVI',
           'NTAP', 'NFLX', 'NWL', 'NFX',
           'NEM', 'NWSA', 'NEE', 'NLSN',
           'NKE', 'NI', 'NE', 'NBL',
           'JWN', 'NSC', 'NTRS', 'NOC',
           'NRG', 'NUE', 'NVDA', 'ORLY',
           'OXY', 'OMC', 'OKE', 'ORCL',
           'OI', 'PCAR', 'PLL', 'PH',
           'PDCO', 'PAYX', 'PNR',
           'PBCT', 'POM', 'PEP',
           'PKI', 'PRGO', 'PFE', 'PCG',
           'PM', 'PSX', 'PNW', 'PXD',
           'PBI', 'PCL', 'PNC', 'RL',
           'PPG', 'PPL', 'PX', 'PCP',
           'PCLN', 'PFG', 'PG', 'PGR',
           'PLD', 'PRU', 'PEG', 'PSA',
           'PHM', 'PVH', 'QRVO', 'PWR',
           'QCOM', 'DGX', 'RRC', 'RTN',
           'O', 'RHT', 'REGN', 'RF',
           'RSG', 'RAI', 'RHI', 'ROK',
           'COL', 'ROP', 'ROST', 'RLC',
           'R', 'CRM', 'SNDK', 'SCG',
           'SLB', 'SNI', 'STX', 'SEE',
           'SRE', 'SHW', 'SIAL', 'SPG',
           'SWKS', 'SLG', 'SJM', 'SNA',
           'SO', 'LUV', 'SWN', 'SE',
           'STJ', 'SWK', 'SPLS', 'SBUX',
            'HOT', 'STT', 'SRCL', 'SYK',
            'STI', 'SYMC', 'SYY', 'TROW',
            'TGT', 'TEL', 'TE', 'TGNA',
            'THC', 'TDC', 'TSO', 'TXN',
            'TXT', 'HSY', 'TRV', 'TMO',
            'TIF', 'TWX', 'TWC', 'TJK',
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
        Get_Stock_Future(symbol)
