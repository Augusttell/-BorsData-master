
# Import local modules
import common
from dataCollection import fetchTickerData
from dataCollection import readFiles
#import models
# import importlib; importlib.reload(dataCollection.fetchTickerData)
# Import PyPi modules
import requests
import json
import numpy as np
import pandas as pd
import os
import re
from random import randrange
from pandas import Series
from matplotlib import pyplot
import matplotlib.dates as mdates
import matplotlib.cbook as cbook


from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
import statsmodels.api as sm

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Global variables
apiKey = "5bd05d51828440c0a88a9ff13f0771a2" # Set API key
currentWD = os.getcwd()

# Initial request TODO move to function(s)
requestInstrument = requests.get("https://apiservice.borsdata.se/v1/instruments?authKey=" + apiKey)
dataInstruments = requestInstrument.json()
sleep(1)
requestCountries = requests.get("https://apiservice.borsdata.se/v1/countries?authKey=" + apiKey)
dataCountries = requestCountries.json()
sleep(1)
requestMarkets = requests.get("https://apiservice.borsdata.se/v1/markets?authKey=" + apiKey)
dataMarkets = requestMarkets.json()
sleep(1)
requestSectors = requests.get("https://apiservice.borsdata.se/v1/sectors?authKey=" + apiKey)
dataSectors = requestSectors.json()
sleep(1)
names_list, urlName_list, instrument_list, ticker_list, sectorId_list, marketId_list, countryId_list, \
insId_list = common.instrumentDictionary(dataInstruments) # Fetch all ticker data


sectorId_list_transl = []
for value in sectorId_list:
    temp = common.getSector(value, dataSectors['sectors'])
    sectorId_list_transl.append(temp)

marketId_list_transl = []
for value in marketId_list:
    temp = common.getMarket(value, dataMarkets['markets'])
    marketId_list_transl.append(temp)

countryId_list_transl = []
for value in countryId_list:
    temp = common.getCountry(value, dataCountries['countries'])
    countryId_list_transl.append(temp)

# Create data folder
try:
    os.mkdir(currentWD + "/data")
except OSError:
    print("Folder already exists")

pd.DataFrame({'Name':names_list, 'Ticker':ticker_list, 'Sector':sectorId_list_transl, 'Market':marketId_list_transl,
              'Country':countryId_list_transl}).to_csv(path_or_buf=currentWD + "/data/Tickers.csv", sep=",",
                                          index=False, decimal=".", encoding="utf-8")

# User defined place of where files are
# TODO Add checks for input file
# TickerfileLocation = input("Please provide file location. \n e.g:  C:/Users/Augus/PycharmProjects/BorsData/testTickers.csv ")
# TickerfileLocation = "C:/Users/Augus/PycharmProjects/BorsData/data/Tickers.csv"
TickerfileLocation = "/Users/august/PycharmProjects/-BorsData-master/data/Tickers.csv"
tickerReadSet = pd.read_csv(filepath_or_buffer=TickerfileLocation)


tickerObjectList = []
i = 0
# TODO make sure todays date is the latest, it is the same
for item in tickerReadSet['Ticker']:
    i = i + 1
    print('Reading item', i, ",", item)
    try:
        tempObject = fetchTickerData(item, ticker_list=ticker_list, insId_list=insId_list, apiKey=apiKey)
        tickerObjectList.append(tempObject)
    except OSError:
        print(item, 'failed collection')
    try:
        os.mkdir(currentWD + "/data/" + item)
        print(item, 'folder created')
    except OSError:
        print(item, 'folder already exists')
    try:
        tempObject.priceData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "price.csv", sep=",",
                                      header=True, index=False, decimal=".")
        print(item, 'price data created')
    except OSError:
        print(item, "couldn't write price data")
    try:
        tempObject.quarterData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "Quarter.csv", sep=",",
                                      header=True, index=False, decimal=".")
        print(item, 'quarter data created')
    except OSError:
        print(item, "couldn't write quarter data")
    try:
        tempObject.yearData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "Year.csv", sep=",",
                                      header=True, index=False, decimal=".")
        print(item, 'year data created')
    except OSError:
        print(item, "couldn't write year data")




# For each in names list call ticker
# TODO add option to exclude data in fetchTickerData, general filter



readData =readFiles(tickerList=tickerReadSet,
               selectedTickers="all",
               selectedSectors="all",
               selectedCountries="all",
               selectedMarket="all"
               )


readData =readFiles(tickerList=tickerReadSet,
               selectedTickers="all",
               selectedSectors="all",
               selectedCountries="all",
               selectedMarket="all"
               )


tickerKeysNew=readData.keys()

# tickerKeysNew
subsetDay = {k:readData[k] for k in [s for s in tickerKeysNew if "price" in s] if k in readData}


def volatilityBreaker(subsetDay, lengthOfData=365, seasonalityLength=31, augustCoefficient = 2, prefix="", breachesNum=50,
                      breachMethod="lower", shiftThreshold = 3, seasonFreq=62):
    stockList = []
    timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')
    try:
        os.mkdir(os.getcwd() + "/" + "volatility_" + prefix + "_" + timeVar + "/")
        print('folder created')
    except OSError:
        print('folder already exists')
    for stock in subsetDay:
        print(stock)
        priceSet=subsetDay[stock][-(lengthOfData + seasonalityLength):]
        if (priceSet['Time'].shape[0] <= 360):
            print("too few observations!")
            continue

        else:
            priceSet.reset_index(inplace=True)
            priceSet['Time'] = pd.to_datetime(priceSet['Time'])
            try:
                result = seasonal_decompose(priceSet.Close,
                                            model='multiplicative',  # multiplicative, additive
                                            freq=seasonFreq,
                                            two_sided=False)
                seasonalEffect = result.seasonal
                trendEffect = result.trend
                stockVar = np.sum(np.power((priceSet.Close - trendEffect), 2))*(1/priceSet.Close.shape[0]) * augustCoefficient

            except OSError:
                print(stock + " failed")
                continue

            data = {'close': priceSet.Close, 'meanValue': trendEffect, 'meanPosVar': trendEffect + stockVar,
                    'meanNegVar': trendEffect - stockVar, 'time': priceSet['Time']}

            # Create DataFrame
            df = pd.DataFrame(data)
            df = df.dropna()
            upperLimitBreach = np.where((df.meanPosVar - df.close) < 0)
            lowerLimitBreach = np.where((df.meanNegVar - df.close) > 0)
            breaches = np.concatenate((upperLimitBreach, lowerLimitBreach), axis=None)

            if(any(df.meanNegVar <0)):
                continue

            if(breachMethod == "upper"):
                breaches=upperLimitBreach[0]
            if(breachMethod == "lower"):
                breaches=lowerLimitBreach[0]

            if(breaches.size >= breachesNum):
                print("too many breaches!")
                continue
            else:
                regressMean = scipy.stats.linregress(x=np.arange(0, df.shape[0]), y=df.close[0:])
                if(regressMean.slope <0):
                    print("Negative trend! ")
                    continue

                else:
                    boolSeq = np.arange(0, df.shape[0]) * regressMean.slope + regressMean.intercept < df.meanValue
                    shift = 0
                    oldState = boolSeq.iloc[0]
                    for val in boolSeq:
                        if(val == True):
                            newState=True
                            if (oldState == False):
                                shift+=1
                                oldState=True
                        if (val == False):
                            newState=False
                            if (oldState == True):
                                shift+=1
                                oldState = False
                    if(shift <= shiftThreshold):
                        print("Too few shifts")
                        continue
                    else:
                        plt.figure(figsize=(14, 8))
                        plotMade = sns.lineplot(x='time', y='value', hue='variable',
                                                data=pd.melt(df, ['time']))
                        plotMade.plot(df["time"].iloc[lowerLimitBreach[0]], df["close"].iloc[lowerLimitBreach[0]], 'g*', c="purple")
                        plotMade.plot(df["time"].iloc[upperLimitBreach[0]], df["close"].iloc[upperLimitBreach[0]], 'g*', c="purple")
                        plotMade.plot(df["time"],np.arange(0, df.shape[0]) * regressMean.slope + regressMean.intercept,ls= '-', c="black", label="Regression line")
                        plt.xticks(rotation=90)
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.title(stock[:-6])
                        plotMade.figure.savefig(
                            os.getcwd() + "/" + "volatility_" + prefix + "_" + timeVar + "/" + stock[:-6] + ".png", bbox_inches='tight')  # save the figure to file
                        # plotMade.plt.close()
                        plt.close(plotMade.figure)
                        tempstock=stock[:-6]
                        stockList.append(tempstock)
    df = pd.DataFrame({'sparade': stockList})
    df.to_csv(path_or_buf=os.getcwd() + "/" + "volatility_" + prefix + "_" + timeVar + "/" + "intressanta.csv", sep=",",
                                  header=True, index=False, decimal=".")

volatilityBreaker(subsetDay=subsetDay, prefix="multiplicative_all", augustCoefficient = 2, seasonFreq=31)

# TODO write a function that plots the stock
# TODO derive a buy price, when should One buy?, Store these create some sort of automated check?
# TODO Function strongest trend of all stocks
# TODO rewrite so only do computations once and they are stored?

np.any("cat" == tickersUnd
# Import local modules
import common
from dataCollection import fetchTickerData
from dataCollection import readFiles
#import models
# import importlib; importlib.reload(dataCollection.fetchTickerData)
# Import PyPi modules
import requests
import json
import numpy as np
import pandas as pd
import os
import re
from random import randrange
from pandas import Series
from matplotlib import pyplot
import matplotlib.dates as mdates
import matplotlib.cbook as cbook


from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
import statsmodels.api as sm

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Global variables
apiKey = "5bd05d51828440c0a88a9ff13f0771a2" # Set API key
currentWD = os.getcwd()

# Initial request TODO move to function(s)
requestInstrument = requests.get("https://apiservice.borsdata.se/v1/instruments?authKey=" + apiKey)
dataInstruments = requestInstrument.json()

requestCountries = requests.get("https://apiservice.borsdata.se/v1/countries?authKey=" + apiKey)
dataCountries = requestCountries.json()

requestMarkets = requests.get("https://apiservice.borsdata.se/v1/markets?authKey=" + apiKey)
dataMarkets = requestMarkets.json()

requestSectors = requests.get("https://apiservice.borsdata.se/v1/sectors?authKey=" + apiKey)
dataSectors = requestSectors.json()

names_list, urlName_list, instrument_list, ticker_list, sectorId_list, marketId_list, countryId_list, \
insId_list = common.instrumentDictionary(dataInstruments) # Fetch all ticker data


sectorId_list_transl = []
for value in sectorId_list:
    temp = common.getSector(value, dataSectors['sectors'])
    sectorId_list_transl.append(temp)

marketId_list_transl = []
for value in marketId_list:
    temp = common.getMarket(value, dataMarkets['markets'])
    marketId_list_transl.append(temp)

countryId_list_transl = []
for value in countryId_list:
    temp = common.getCountry(value, dataCountries['countries'])
    countryId_list_transl.append(temp)

# Create data folder
try:
    os.mkdir(currentWD + "/data")
except OSError:
    print("Folder already exists")

pd.DataFrame({'Name':names_list, 'Ticker':ticker_list, 'Sector':sectorId_list_transl, 'Market':marketId_list_transl,
              'Country':countryId_list_transl}).to_csv(path_or_buf=currentWD + "/data/Tickers.csv", sep=",",
                                          index=False, decimal=".", encoding="utf-8")

# User defined place of where files are
# TODO Add checks for input file
# TickerfileLocation = input("Please provide file location. \n e.g:  C:/Users/Augus/PycharmProjects/BorsData/testTickers.csv ")
# TickerfileLocation = "C:/Users/Augus/PycharmProjects/BorsData/data/Tickers.csv"
TickerfileLocation = "/Users/august/PycharmProjects/-BorsData-master/data/Tickers.csv"
tickerReadSet = pd.read_csv(filepath_or_buffer=TickerfileLocation)


tickerObjectList = []
i = 0
# TODO make sure todays date is the latest, it is the same
for item in tickerReadSet['Ticker']: # TODO make this a method to store all instruments locally
    i = i + 1
    print('Reading item', i, ",", item)
    try:
        tempObject = fetchTickerData(item, ticker_list=ticker_list, insId_list=insId_list, apiKey=apiKey)
        tickerObjectList.append(tempObject)
    except OSError:
        print(item, 'failed collection')
    try:
        os.mkdir(currentWD + "/data/" + item)
        print(item, 'folder created')
    except OSError:
        print(item, 'folder already exists')
    try:
        tempObject.priceData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "price.csv", sep=",",
                                      header=True, index=False, decimal=".")
        print(item, 'price data created')
    except OSError:
        print(item, "couldn't write price data")
    try:
        tempObject.quarterData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "Quarter.csv", sep=",",
                                      header=True, index=False, decimal=".")
        print(item, 'quarter data created')
    except OSError:
        print(item, "couldn't write quarter data")
    try:
        tempObject.yearData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "Year.csv", sep=",",
                                      header=True, index=False, decimal=".")
        print(item, 'year data created')
    except OSError:
        print(item, "couldn't write year data")




# For each in names list call ticker
# TODO add option to exclude data in fetchTickerData, general filter



readData =readFiles(tickerList=tickerReadSet,
               selectedTickers="all",
               selectedSectors="all",
               selectedCountries="all",
               selectedMarket="all"
               )


readData =readFiles(tickerList=tickerReadSet,
               selectedTickers="all",
               selectedSectors="all",
               selectedCountries="all",
               selectedMarket="all"
               )


tickerKeysNew=readData.keys()

# tickerKeysNew
subsetDay = {k:readData[k] for k in [s for s in tickerKeysNew if "price" in s] if k in readData}




volatilityBreaker(subsetDay=subsetDay, prefix="multiplicative_all", augustCoefficient = 2, seasonFreq=31)

# TODO write a function that plots the stock

# TODO Function strongest trend of all stocks, momentum investing







tickersUnderStudy = pd.read_csv(filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/volatility_multiplicative_all_2020_01_05_20/intressanta.csv")


stateSpaceModel(subsetDay, tickers=tickersUnderStudy[1:5])







# We could further add a damped stochastic cycle as follows
# mod_cycle = sm.tsa.UnobservedComponents(subsetDay["TELIA_price"]["Close"][-365:],
#                                         level='lldtrend',#strend, llevel, lldtrend
#                                         trend=True,
#                                         seasonal=61,
#                                         # cycle=True,
#                                         # autoregressive=3,
#                                         stochastic_level=True,
#                                         # ,
#                                         # ,
#                                         irregular = True
#                                         # damped_cycle=True,
#                                         # stochastic_cycle=False,
#                                         )
# res_cycle = mod_cycle.fit()

# Create plots with confidence intervals:
# observed vs predicted
# predictionsData = res_cycle.filter_results.forecasts[0]
# std_errors = np.sqrt(res_cycle.filter_results.forecasts_error_cov[0, 0])
# alpha=0.05
# critical_value = norm.ppf(1 - alpha / 2.)
# ci_lower = predictionsData - critical_value * std_errors
# ci_upper = predictionsData + critical_value * std_errors
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'observed': subsetDay["TELIA_price"]["Close"][-365:], 'predicted': predictionsData,
#         'lower': ci_lower, 'upper': ci_upper}
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8))
# plotMade = sns.lineplot(x='Time', y='observed', data=df[-300:], label='Observed')
# plotMade.plot(df.Time[-300:], df.predicted[-300:], color='k',
#                     label='Predicted')
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Predictes vs observed "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)

# level
# levelData = res_cycle.level["smoothed"]
# std_errors = np.sqrt(res_cycle.level["smoothed_cov"])
# ci_lower = levelData - critical_value * std_errors
# ci_upper = levelData + critical_value * std_errors
#
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'Level': levelData,
#         'lower': ci_lower, 'upper': ci_upper}
#
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8))
# plotMade = sns.lineplot(x='Time', y='Level', data=df[-300:], label='Level')
#
#
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Level "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)

# # Trend
# trendData = res_cycle.trend["smoothed"]
# std_errors = np.sqrt(res_cycle.trend["smoothed_cov"])
# ci_lower = trendData - critical_value * std_errors
# ci_upper = trendData + critical_value * std_errors
#
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'Trend': trendData,
#         'lower': ci_lower, 'upper': ci_upper}
#
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8))
# plotMade = sns.lineplot(x='Time', y='Trend', data=df[-300:], label='Trend')
#
#
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Trend "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)
#
# # Seasonal
# seasonalData = res_cycle.seasonal["smoothed"]
# std_errors = np.sqrt(res_cycle.seasonal["smoothed_cov"])
# ci_lower = seasonalData - critical_value * std_errors
# ci_upper = seasonalData + critical_value * std_errors
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'Seasonal': seasonalData,
#         'lower': ci_lower, 'upper': ci_upper}
#
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8)
# plotMade = sns.lineplot(x='Time', y='Seasonal', data=df[-300:], label='Seasonal')
#
#
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Seasonal "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)



# # Forecast
# predForeCast  = res_cycle.forecast(steps=61, )
# plt.figure(figsize=(14, 8))
#
# plotMade = sns.lineplot(x=subsetDay["TELIA_price"]["Close"][-365:])
# plotMade.plot(x = np.arange(predForeCast.index[0],predForeCast.index[60]), y=predForeCast,  c="purple")
#
# meanObserved = subsetDay["TOURN_price"]["Close"][-10:].mean()
# forecastValues = res_cycle.forecast(steps=61)
# meanForecast  = round(np.mean(forecastValues[-10:]), 2)
#
# if((meanObserved/meanForecast) >= 1.05):
#
#
#
#
# plotMade = sns.lineplot(data=subsetDay["TELIA_price"]["Close"][-365:])
# plotMade.plot(res_cycle.forecast(steps=61), '-', c="purple")
#
#
#
#
#
#
# np.arange(predForeCast.index[0],predForeCast.index[60])
#
#
# subsetDay["TELIA_price"]["Time"][-365:].shape[0]
#
#
#
# # Decide some general rule such as: mean of last 10 observed observations should be x time smaller than mean of last 20?
# # Always produce this output seen above and create reoo
# # Lastly create a list of stock desicions decision. suggested buy price and sufggested sell price
#
#
# erStudy)





# def stateSpaceModel(subsetDay, lengthOfData=365, seasonalityLength=31, prefix="", threshold=1.1, tickers=tickersUnderStudy):
#     buyPrice = []
#     sellPrice = []
#     decision = []
#     stockList = []
#     timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')
#     try:
#         os.mkdir(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/")
#         print('folder created')
#     except OSError:
#         print('folder already exists')
#
#     for stock in subsetDay:
#         tickers
#         stockList.append(stock[:-6])
#
#         try:
#             os.mkdir(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock + "/")
#             print('folder created for ' + stock)
#         except OSError:
#             print('folder for stock '+ stock +  ' already exists')
#
#         priceSet = subsetDay[stock][-(lengthOfData + seasonalityLength):]
#         mod_cycle = sm.tsa.UnobservedComponents(priceSet["Close"],
#                                                 level='lldtrend',  # strend, llevel, lldtrend
#                                                 trend=True,
#                                                 seasonal=seasonalityLength,
#                                                 stochastic_level=True,
#                                                 irregular=True)
#         res_cycle = mod_cycle.fit()
#
#         # observed vs predicted
#         predictionsData = res_cycle.filter_results.forecasts[0]
#         std_errors = np.sqrt(res_cycle.filter_results.forecasts_error_cov[0, 0])
#         alpha = 0.05
#         critical_value = norm.ppf(1 - alpha / 2.)
#         ci_lower = predictionsData - critical_value * std_errors
#         ci_upper = predictionsData + critical_value * std_errors
#
#         data = {'Time': priceSet["Time"], 'observed': priceSet["Close"],
#                 'predicted': predictionsData,
#                 'lower': ci_lower, 'upper': ci_upper}
#
#         df = pd.DataFrame(data)
#         df.reset_index(inplace=True)
#         df['Time'] = pd.to_datetime(df['Time'])
#         plt.figure(figsize=(14, 8))
#         plotMade = sns.lineplot(x='Time', y='observed', data=df[-300:], label='Observed')
#         plotMade.plot(df.Time[-300:], df.predicted[-300:], color='k',
#                       label='Predicted')
#         plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title("Predictes vs observed " + stock[:-6])
#         plt.xticks(rotation=90)
#         plotMade.figure.savefig(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
#                                 "/predVsObs.png", bbox_inches='tight')
#         plt.close(plotMade.figure)
#
#
#         # level
#         levelData = res_cycle.level["smoothed"]
#         std_errors = np.sqrt(res_cycle.level["smoothed_cov"])
#         ci_lower = levelData - critical_value * std_errors
#         ci_upper = levelData + critical_value * std_errors
#
#         data = {'Time': priceSet["Time"], 'Level': levelData,
#                 'lower': ci_lower, 'upper': ci_upper}
#
#         df = pd.DataFrame(data)
#         df.reset_index(inplace=True)
#         df['Time'] = pd.to_datetime(df['Time'])
#         plt.figure(figsize=(14, 8))
#         plotMade = sns.lineplot(x='Time', y='Level', data=df[-300:], label='Level')
#
#         plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title("Level " + stock[:-6])
#         plt.xticks(rotation=90)
#         plotMade.figure.savefig(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
#                                 "/level.png", bbox_inches='tight')
#         plt.close(plotMade.figure)
#
#         # Trend
#         trendData = res_cycle.trend["smoothed"]
#         std_errors = np.sqrt(res_cycle.trend["smoothed_cov"])
#         ci_lower = trendData - critical_value * std_errors
#         ci_upper = trendData + critical_value * std_errors
#
#         data = {'Time': priceSet["Time"], 'Trend': trendData,
#                 'lower': ci_lower, 'upper': ci_upper}
#
#         df = pd.DataFrame(data)
#         df.reset_index(inplace=True)
#         df['Time'] = pd.to_datetime(df['Time'])
#         plt.figure(figsize=(14, 8))
#         plotMade = sns.lineplot(x='Time', y='Trend', data=df[-300:], label='Trend')
#
#         plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title("Trend " + stock[:-6])
#         plt.xticks(rotation=90)
#         plotMade.figure.savefig(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
#                                 "/trend.png", bbox_inches='tight')
#         plt.close(plotMade.figure)
#
#         # Seasonal
#         seasonalData = res_cycle.seasonal["smoothed"]
#         std_errors = np.sqrt(res_cycle.seasonal["smoothed_cov"])
#         ci_lower = seasonalData - critical_value * std_errors
#         ci_upper = seasonalData + critical_value * std_errors
#
#         data = {'Time': priceSet["Time"], 'Seasonal': seasonalData,
#                 'lower': ci_lower, 'upper': ci_upper}
#
#         df = pd.DataFrame(data)
#         df.reset_index(inplace=True)
#         df['Time'] = pd.to_datetime(df['Time'])
#         plt.figure(figsize=(14, 8))
#         plotMade = sns.lineplot(x='Time', y='Seasonal', data=df[-300:], label='Seasonal')
#
#         plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
#         plt.xlabel('Time')
#         plt.ylabel('Price')
#         plt.title("Seasonal " + stock[:-6])
#         plt.xticks(rotation=90)
#         plotMade.figure.savefig(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
#                                 "/seasonal.png", bbox_inches='tight')
#         plt.close(plotMade.figure)
#
#         # Forecast
#         # predictedResults = res_cycle.get_prediction(start=priceSet["Close"].shape[0] - 31,
#         #                                             end=priceSet["Close"].shape[0] + 30, dynamic=False)
#         #
#         # predictedResults = res_cycle.get_prediction(start=priceSet["Close"].shape[0]-30,
#         #                                             end=priceSet["Close"].shape[0]+32, dynamic=True)
#         #
#         # predictedResults = res_cycle.get_prediction(start=priceSet["Close"].index[0]+1,
#         #                                             end=priceSet["Close"].index[0]+21, dynamic=True)
#
#         predict_df=res_cycle.forecast(62)
#
#         # predict_df = predictedResults.summary_frame(alpha=0.1)
#
#         priceSet['Time'] = pd.to_datetime(priceSet['Time'])
#         temp = priceSet["Time"].iloc[-1]
#         t = datetime.timedelta(days=1)
#         foreCastStartDate = temp + t
#         predict_df.index = pd.date_range(start=foreCastStartDate, periods=62, freq='D')
#         plotMade = sns.lineplot(
#             y=priceSet["Close"], x=priceSet["Time"])
#         plotMade.plot(predict_df, c="purple")
#
#
#         plt.xticks(rotation=90)
#         plotMade.figure.savefig(os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
#                                 "/prediction.png", bbox_inches='tight')
#         plt.close(plotMade.figure)
#
#         meanObserved = priceSet["Close"][-10:].mean()
#         meanForecast = round(np.mean(predict_df[-20:]), 2)
#
#         temp1=meanObserved
#         temp2=meanForecast
#         buyPrice.append(temp1)
#         sellPrice.append(temp2)
#         decisionTemp="No"
#         if ((meanObserved / meanForecast) >= threshold):
#             decisionTemp = "Yes"
#
#         decision.append(decisionTemp)
#
#     df = pd.DataFrame({'Aktier': stockList, "KöprPris":buyPrice, "SäljPris":sellPrice, "beslut":decision})
#     df.to_csv(path_or_buf=os.getcwd() + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + "/aktier.csv",
#               sep=",", header=True, index=False, decimal=".")




tickersUnderStudy = pd.read_csv(filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/volatility_multiplicative_all_2020_01_05_20/intressanta.csv")


stateSpaceModel(subsetDay, tickers=tickersUnderStudy[1:3])







# We could further add a damped stochastic cycle as follows
# mod_cycle = sm.tsa.UnobservedComponents(subsetDay["TELIA_price"]["Close"][-365:],
#                                         level='lldtrend',#strend, llevel, lldtrend
#                                         trend=True,
#                                         seasonal=61,
#                                         # cycle=True,
#                                         # autoregressive=3,
#                                         stochastic_level=True,
#                                         # ,
#                                         # ,
#                                         irregular = True
#                                         # damped_cycle=True,
#                                         # stochastic_cycle=False,
#                                         )
# res_cycle = mod_cycle.fit()

# Create plots with confidence intervals:
# observed vs predicted
# predictionsData = res_cycle.filter_results.forecasts[0]
# std_errors = np.sqrt(res_cycle.filter_results.forecasts_error_cov[0, 0])
# alpha=0.05
# critical_value = norm.ppf(1 - alpha / 2.)
# ci_lower = predictionsData - critical_value * std_errors
# ci_upper = predictionsData + critical_value * std_errors
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'observed': subsetDay["TELIA_price"]["Close"][-365:], 'predicted': predictionsData,
#         'lower': ci_lower, 'upper': ci_upper}
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8))
# plotMade = sns.lineplot(x='Time', y='observed', data=df[-300:], label='Observed')
# plotMade.plot(df.Time[-300:], df.predicted[-300:], color='k',
#                     label='Predicted')
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Predictes vs observed "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)

# level
# levelData = res_cycle.level["smoothed"]
# std_errors = np.sqrt(res_cycle.level["smoothed_cov"])
# ci_lower = levelData - critical_value * std_errors
# ci_upper = levelData + critical_value * std_errors
#
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'Level': levelData,
#         'lower': ci_lower, 'upper': ci_upper}
#
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8))
# plotMade = sns.lineplot(x='Time', y='Level', data=df[-300:], label='Level')
#
#
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Level "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)

# # Trend
# trendData = res_cycle.trend["smoothed"]
# std_errors = np.sqrt(res_cycle.trend["smoothed_cov"])
# ci_lower = trendData - critical_value * std_errors
# ci_upper = trendData + critical_value * std_errors
#
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'Trend': trendData,
#         'lower': ci_lower, 'upper': ci_upper}
#
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8))
# plotMade = sns.lineplot(x='Time', y='Trend', data=df[-300:], label='Trend')
#
#
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Trend "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)
#
# # Seasonal
# seasonalData = res_cycle.seasonal["smoothed"]
# std_errors = np.sqrt(res_cycle.seasonal["smoothed_cov"])
# ci_lower = seasonalData - critical_value * std_errors
# ci_upper = seasonalData + critical_value * std_errors
#
# data = {'Time': subsetDay["TELIA_price"]["Time"][-365:], 'Seasonal': seasonalData,
#         'lower': ci_lower, 'upper': ci_upper}
#
# df = pd.DataFrame(data)
# df.reset_index(inplace=True)
# df['Time'] = pd.to_datetime(df['Time'])
# plt.figure(figsize=(14, 8)
# plotMade = sns.lineplot(x='Time', y='Seasonal', data=df[-300:], label='Seasonal')
#
#
# plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:],alpha=0.2)
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title("Seasonal "+ stock[:-6])
# plt.xticks(rotation=90)
# plt.close(plotMade.figure)


import scipy

nbinom=scipy.stats.nbinom.rvs(1, 0.003, loc=0, size=2000, random_state=None)


import matplotlib.pyplot as plt
plt.hist(nbinom, bins = 50)