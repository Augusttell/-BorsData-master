import os

import pandas as pd
import datetime

import dataCollection as dc
import models


# TODO add timer
# TODO add email
# TODO ADD CYCLE EFFECT FOR ALL ?
# TODO ADJUST VOLATILITY with volumev
# TODO add back testing?

# Global variables
apiKey = "5bd05d51828440c0a88a9ff13f0771a2"  # Set API key
currentWD = os.getcwd()

# Collect ticker metadata
names_list, urlName_list, instrument_list, ticker_list, sectorId_list, marketId_list, countryId_list, \
insId_list = dc.collect_ticker_metadata(apiKey, currentWD)

TickerfileLocation = "/Users/august/PycharmProjects/-BorsData-master/data/Tickers.csv"
tickerReadSet = pd.read_csv(filepath_or_buffer=TickerfileLocation)

# Fetch all ticker data
dc.download_all_data(tickerReadSet=tickerReadSet, ticker_list=ticker_list, insId_list=insId_list,
                     apiKey=apiKey, currentWD=currentWD)

readData = dc.read_files_from_disk(tickerList=tickerReadSet,
                                   selectedTickers=["AAK", "OMXSPI"],
                                   selectedSectors="all",
                                   selectedCountries="all",
                                   selectedMarket="all")

tickerKeysNew = readData.keys()
# tickerKeysNew
subsetDay: dict = {k: readData[k] for k in [s for s in tickerKeysNew if "price" in s] if k in readData}

timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')
# timeVar = "2020_01_09_20"
try:
    os.mkdir(os.getcwd() + "/" + timeVar)
    print('folder created')
except OSError:
    print('folder already exists')


# # Quarter Swing
# # Good tuned
# quarterVolatilityPath = models.volatilityBreaker(subsetDay=subsetDay, repo=timeVar, prefix="quarter",
#                                                  augustCoefficient=2,
#                                                  seasonFreq=93, breachesNum=50, lengthOfData=365, seasonalityLength=93,
#                                                  breachMethod="lower", shiftThreshold=4)
#
# quarterTickersUnderStudy = pd.read_csv(filepath_or_buffer=quarterVolatilityPath)
# models.stateSpaceModel(subsetDay, tickersUnderStudy=quarterTickersUnderStudy, lengthOfData=365, seasonalityLength=93,
#                        repo=timeVar, prefix="quarter", threshold=1.1, foreCastLength=93)
#
# # Two month swing
# # Good tuned
# twoMonthVolatilityPath = models.volatilityBreaker(subsetDay=subsetDay, repo=timeVar, prefix="twoMonth",
#                                                   augustCoefficient=2,
#                                                   seasonFreq=62, breachesNum=50, lengthOfData=365, seasonalityLength=62,
#                                                   breachMethod="lower", shiftThreshold=4)
#
# twoMonthTickersUnderStudy = pd.read_csv(filepath_or_buffer=twoMonthVolatilityPath)
# models.stateSpaceModel(subsetDay, lengthOfData=365, seasonalityLength=62, repo=timeVar, prefix="twoMonth",
#                        threshold=1.1,
#                        tickersUnderStudy=twoMonthTickersUnderStudy, foreCastLength=62)
#
# # One month swing
# # Good tuned
# oneMonthVolatilityPath = models.volatilityBreaker(subsetDay=subsetDay, repo=timeVar, prefix="oneMonth",
#                                                   augustCoefficient=0.5,
#                                                   seasonFreq=31, breachesNum=25, lengthOfData=365, seasonalityLength=31,
#                                                   breachMethod="lower", shiftThreshold=4)
#
# oneMonthTickersUnderStudy = pd.read_csv(filepath_or_buffer=oneMonthVolatilityPath)
# # oneMonthTickersUnderStudy = pd.read_csv(filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/2020_01_08_21/volatility_oneMonth_2020_01_08_22/intressanta.csv")
# models.stateSpaceModel(subsetDay, lengthOfData=365, seasonalityLength=31, repo=timeVar, prefix="oneMonth",
#                        threshold=1.02, includeCycle=True,
#                        tickersUnderStudy=oneMonthTickersUnderStudy, foreCastLength=31)

# Two week swing
# Good tuned
# twoWeekVolatilityPath = models.volatilityBreaker(subsetDay=subsetDay, repo=timeVar, prefix="twoWeek",
#                                                  augustCoefficient=.25,
#                                                  seasonFreq=14, breachesNum=50, lengthOfData=365, seasonalityLength=14,
#                                                  breachMethod="lower", shiftThreshold=2)
#
# twoWeekTickersUnderStudy = pd.read_csv(filepath_or_buffer=twoWeekVolatilityPath)
# # twoWeekTickersUnderStudy = pd.read_csv(filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/2020_01_09_20/volatility_twoWeek_2020_01_09_20/intressanta.csv")
# models.stateSpaceModel(subsetDay, lengthOfData=365, seasonalityLength=14, repo=timeVar, prefix="twoWeek",
#                        threshold=1.05, includeCycle=True,
#                        tickersUnderStudy=twoWeekTickersUnderStudy, foreCastLength=14)

# One week swing
# Good tuned
oneWeekVolatilityPath = models.volatilityBreaker(subsetDay=subsetDay, repo=timeVar, prefix="oneWeek",
                                                 augustCoefficient=.75,
                                                 seasonFreq=7, breachesNum=50, lengthOfData=365, seasonalityLength=7,
                                                 breachMethod="lower", shiftThreshold=0)
oneWeekTickersUnderStudy = pd.read_csv(filepath_or_buffer=oneWeekVolatilityPath)
models.stateSpaceModel(subsetDay, lengthOfData=365, seasonalityLength=7, repo=timeVar, prefix="oneWeek",
                       threshold=1.05, includeCycle=True,
                       tickersUnderStudy=oneWeekTickersUnderStudy, foreCastLength=7)

