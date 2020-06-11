import os
import pandas as pd

from scipy.optimize import linprog
import numpy as np
import dataCollection as dc
import preProcessing as prep

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense

# TODO Change all functions to fit Pep8 naming conventions
# TODO write main function
# TODO add function descriptions


def main():
    # TODO implement a proper main
 #https://www.reddit.com/r/computersciencehub/comments/gg9scz/how_to_use_main_function_in_python_with_example/?utm_source=share&utm_medium=ios_app&utm_name=iossmf

# Global variables
apiKey = "5bd05d51828440c0a88a9ff13f0771a2"  # Set API key
currentWD = os.getcwd()

# Collect ticker metadata
names_list, urlName_list, instrument_list, ticker_list, sectorId_list, marketId_list, countryId_list, \
insId_list = dc.collect_ticker_metadata(apiKey, currentWD)

tickerfileLocation = "/Users/august/PycharmProjects/-BorsData-master/data/Tickers.csv"
tickerReadSet = pd.read_csv(filepath_or_buffer=tickerfileLocation)

# Fetch all ticker data
dc.download_all_data(tickerReadSet=tickerReadSet, ticker_list=ticker_list, insId_list=insId_list,
                     apiKey=apiKey, currentWD=currentWD)

tickerReadSet = tickerReadSet.drop(tickerReadSet[tickerReadSet.Ticker == "HUDYA"].index)
tickerReadSet = tickerReadSet.drop(tickerReadSet[tickerReadSet.Ticker == "PRIV"].index)
tickerReadSet = tickerReadSet.drop(tickerReadSet[tickerReadSet.Ticker == "VOLAB"].index)
tickerReadSet = tickerReadSet.drop(tickerReadSet[tickerReadSet.Ticker == "QLIFE"].index)

readData = dc.read_files_from_disk(tickerList=tickerReadSet,
                                   selectedTickers="all",
                                   selectedSectors="all",
                                   selectedCountries="all",
                                   selectedMarket="all")

tickerKeysNew = readData.keys()

quarter_data: dict = {k: readData[k] for k in [s for s in tickerKeysNew if "quarter" in s] if k in readData}
price_data: dict = {k: readData[k] for k in [s for s in tickerKeysNew if "price" in s] if k in readData}

del tickerKeysNew
del readData
# TODO Merge this loop with above
for stock in quarter_data.copy():
    length = quarter_data[stock].shape[0]
    if length < 5:
        del quarter_data[stock]
        continue
    if length - len(np.where(quarter_data[stock].stock_Price_High == 0)[0]) < 5:
        del quarter_data[stock]
        continue
    if length - len(np.where(quarter_data[stock].stock_Price_High == 0)[0]) < 5:
        del quarter_data[stock]
        continue


# Prepp data
quarter_data_frame = prep.prepare_quarter_data(quarter_data)
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Prelist'].Ticker)]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Pepins Market'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Index'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Spotlight'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Merkur Market'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='OB Standard'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Oslo Axess'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='OBX'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Oslo Axess'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='OB Match'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='NGM'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='First North'].Ticker+"_quarter")]
quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Small Cap'].Ticker+"_quarter")]




# quarter_data_frçame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market!='Large Cap' ].Ticker+"_quarter")]

# quarter_data_frame = quarter_data_frame[~quarter_data_frame['stock'].isin(tickerReadSet[tickerReadSet.Market=='Index'].Ticker+"_quarter")]

quarter_data["ABB_quarter"].report_End_Date
quarter_data["ABB_quarter"].stock_Price_Average



price_data["ABB_price"][(price_data["ABB_price"].Time >= "2019-09-30") & (price_data["ABB_price"].Time < "2019-12-30")].Close.mean()

quarter_data_frame = prep.prepare_quarter_data(quarter_data)
quarter_list = quarter_data["ABB_quarter"].report_End_Date

extract, mean median std of each attribute during each period
Index(['Time', 'Close', 'Open', 'High', 'Low', 'volume'], dtype='object')


def prepare_price_for_quarter_data(price_data, quarter_list):
    i = 0
    for stock in price_data:
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Close.mean()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Close.median()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Close.std()

        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Open.mean()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Open.median()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Open.std()

        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].High.mean()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].High.median()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].High.std()

        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Low.mean()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Low.median()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].Low.std()

        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].volume.mean()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].volume.median()
        price_data[stock][
            (price_data[stock].Time >= quarter_list[1]) & (price_data[stock].Time < quarter_list[0])].volume.std()







tickerReadSet[tickerReadSet.Ticker=="GIGSEK"]


data = modelRNN(quarter_data_frame, seq_length = 5)


data.train_model(epochs=1000, layers=250)
predicted_holdout=data.predict_holdout(data.holdout_feature_array)

# cvarList = riskMeasurements(price_data, h=5, tradingDays=252, alpha=0.01)
model = Sequential()
# model.add(LSTM(250, activation='relu', input_shape=(4, data.train_feature_array.shape[2])))
model.add(LSTM(250, activation='relu', return_sequences=True, input_shape=(4, data.train_feature_array.shape[2])))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
# model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
history = model.fit(data.train_feature_array, data.train_target_array, epochs=100,
                              validation_split=0.2, verbose=1, batch_size=10)

predicted_holdout = model.predict(data.holdout_feature_array, verbose=1)
import matplotlib.pyplot as plt
import matplotlib.lines as lines

f = plt.figure()

plt.xlabel('Predicted')
plt.ylabel('Observed')
s = plt.scatter(x=predicted_holdout, y=data.holdout_target_array)
x = np.linspace(-.5, .5, 1000)
plt.plot(x, x + 0, linestyle='solid')
plt.vlines(x=0, color = "red", xmin=1,xmax=1)
plt.hlines(y=0, color = "red", ymin=1,ymax=1)


# TODO add evaluation
# TODO add feature for quarter
# TODO add feature for average trade volume
# TODO TRY longer and shorter sequence, (longer seem to be working)
# TODO add volatility as a feature
# TODO try and predict price and not returns
np.mean(np.abs(predicted_holdout - data.holdout_target_array))
len(np.where(predicted_holdout-data.holdout_target_array >0)) # Over predict returns
len(predicted_holdout-data.holdout_target_array >0) # Under predict returns


# from scipy import stats
# slope, intercept, r_value, p_value, std_err = stats.linregress(np.reshape(predicted_holdout, predicted_holdout.shape[0]), data.holdout_target_array[0:])
# line = slope*predicted_holdout+intercept
# plt.plot(predicted_holdout, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
#
# Holdout set


import numpy as np
import matplotlib.cm as cm


class modelRNN():
    def __init__(self, quarter_data, seq_length = 4, holdout_date = "2019-09-30"):

        # Handle NA's
        quarter_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        quarter_data = quarter_data.fillna(0)

        # Normalization
        quarter_data[quarter_data.columns.difference(['report_End_Date', 'returns', "stock"])] = \
            (quarter_data[quarter_data.columns.difference(['report_End_Date', 'returns', "stock"])] - \
             quarter_data[quarter_data.columns.difference(['report_End_Date', 'returns', "stock"])].min()) / \
            (quarter_data[quarter_data.columns.difference(['report_End_Date', 'returns', "stock"])].max() -
             quarter_data[quarter_data.columns.difference(['report_End_Date', 'returns', "stock"])].min())

        # Make data ready for RNN, many-to-one problem
        i=0
        for stock in quarter_data.stock.unique():
            data_iteration = quarter_data[quarter_data['stock'] == stock]
            feature_array = np.array(
                data_iteration[data_iteration.columns.difference([#'report_End_Date',
                                                                  #'returns',
                    "stock"
                                                                  ])])
            target_array = np.array(data_iteration['returns'])
            num_samples = target_array.shape[0]-seq_length

            feature_array2=np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)

            feature_array_reshaped = np.array([np.stack(feature_array2[i:i + 4]) for i in range(1, num_samples + 1)])
            target_array_reshaped = target_array[0:num_samples]

            if i == 0:
                full_feature_array = feature_array_reshaped
                full_target_array = target_array_reshaped
            else:
                full_feature_array = np.concatenate((full_feature_array, feature_array_reshaped))
                full_target_array = np.concatenate((full_target_array, target_array_reshaped))
            i = +1

        # Split holdout and train set
        holdout_index = np.where(full_feature_array[:, 0, 47] >= holdout_date)
        train_index = np.where(full_feature_array[:, 0, 47] < holdout_date)

        self.holdout_feature_array = full_feature_array[holdout_index, :, ][0]
        self.holdout_feature_array = np.delete(self.holdout_feature_array, 47, axis=2)

        self.train_feature_array = full_feature_array[train_index, :, :][0]
        self.train_feature_array = np.delete(self.train_feature_array, 47, axis=2)

        self.holdout_target_array = full_target_array[holdout_index]
        self.train_target_array = full_target_array[train_index]

    def train_model(self, seq_length = 4, epochs=10, layers=50):
        # TODO add cross validation
        self.model = Sequential()
        self.model.add(LSTM(layers, activation='relu', input_shape=(seq_length, self.train_feature_array.shape[2])))
        self.model.add(Dense(1))


        # model = Sequential()
        # model.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(3, 1)))
        # model.add(LSTM(100, activation='relu', return_sequences=True))
        # model.add(LSTM(50, activation='relu', return_sequences=True))
        # model.add(LSTM(25, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        # model.add(Dense(10, activation='relu'))
        # model.add(Dense(1))


        self.model.compile(optimizer='adam', loss='mae')
        self.history = self.model.fit(self.train_feature_array, self.train_target_array, epochs=epochs,
                                      validation_split=0.2, verbose=1)
        return(self.history)

    def predict_holdout(self, test_input):
        self.test_output = self.model.predict(test_input, verbose=1)
        return(self.test_output)








from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional

# TODO Train model method, how much data as input?
# TODO Evaluate model method
# TODO Predict quarter







import scipy.stats as ss
import numpy as np

class riskMeasurements():
    def __init__(self, price_data, h, tradingDays=252, alpha=0.01, distribution="normal"):
        # TODO Add another distribution
        # TODO Add plots methods
        i = 0
        for stock in price_data:
            returns = price_data[stock]["Close"].pct_change(1)
            returns = returns.drop(0)
            sigma = returns.tail(h).std()
            mu = returns.tail(h).mean()
            sigma_h = sigma * np.sqrt(h / tradingDays)
            cVaR_stock = alpha ** -1 * ss.norm.pdf(ss.norm.ppf(alpha)) * sigma_h - mu
            d = {'cVaR_h_' + str(h) + "_td_" + str(tradingDays) + "_alph_" + str(alpha) + "_dist_" + distribution: [
                cVaR_stock],
                 'stock': stock}
            cvar_dataframe = pd.DataFrame(data=d)
            if i == 0:
                cvar_dataframe_all = cvar_dataframe
            if i > 0:
                cvar_dataframe_all = pd.concat([cvar_dataframe_all, cvar_dataframe], axis=0)
            i = +1

        self.cvar_dataframe = cvar_dataframe_all


cvarList = riskMeasurements(price_data, h=5, tradingDays=252, alpha=0.01)

# normal distribution
cvarList.cvar_dataframe

       # h 5 day
            # h 14 Day
            # h 31 Day
            # h 62 Day
            # h 93 day


    # http: // www.quantatrisk.com / 2016 / 12 / 0
    # 8 / conditional - value - at - risk - normal - student - t - var - model - python /
    # Var
    # Compute cvar for multiple distributions
        # Plot results
# https://en.wikipedia.org/wiki/Expected_shortfall
    #






def inflationAdjuster():
    # TODO adjust for inflation



def currency_translator():
    # Todo adjust for currencies





class portfolioOptimization():
    # TODO maybe add mean value optimization here as an option instead of duality programming

    def efficient_frontier(c, A_ub, b_ub, A_eq, b_eq, bounds):
        # TODO turn in to method
        # TODO add method of generate weights
        # Return an efficient frontier
        results = linprog(c=returnsEx,
                          A_ub=None,
                          b_ub=None,
                          A_eq=[amtL],
                          b_eq=[1],
                          bounds=(0, 1),
                          method='interior-point',
                          callback=None,
                          options=None,
                          x0=None)
        return results







