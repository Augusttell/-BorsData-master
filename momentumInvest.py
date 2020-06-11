import os
import pandas as pd
import datetime

import dataCollection as dc
from models import momentum


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
                                   selectedTickers="all",
                                   selectedSectors="all",
                                   selectedCountries="all",
                                   selectedMarket="all")

tickerKeysNew = readData.keys()
# tickerKeysNew
subsetDay: dict = {k: readData[k] for k in [s for s in tickerKeysNew if "price" in s] if k in readData}

timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')

try:
    os.mkdir(os.getcwd() + "/" + timeVar)
    print('folder created')
except OSError:
    print('folder already exists')


# Momentum investing
# momentumPath = momentum(subsetDay=subsetDay,
#                       lengthOfData = 1095,
#                       seasonalityLength= 93,
#                       microLengthRegression= 62,
#                       repo = timeVar,
#                       prefix = "threeYear_13",
#                       varMulti=1.3,
#                       breachesNum = 1,
#                       seasonFreq = 93,
#                       longThresholdMulti = 3, #
#                       shortThreshold= 1.02,
#                       tradingDays = 252,
#                       BetaIndex = "OMXSPI_price")
#
# momentumPath = momentum(subsetDay=subsetDay,
#                       lengthOfData = 730,
#                       seasonalityLength= 93,
#                       microLengthRegression= 62,
#                       repo = timeVar,
#                       prefix = "twoYear_13",
#                       varMulti=1.3,
#                       breachesNum = 1,
#                       seasonFreq = 93,
#                       longThresholdMulti = 3, #
#                       shortThreshold= 1.02,
#                       tradingDays = 252,
#                       BetaIndex = "OMXSPI_price")

momentumPath = momentum(subsetDay=subsetDay,
                      lengthOfData = 365,
                      seasonalityLength= 93,
                      microLengthRegression= 62,
                      repo = timeVar,
                      prefix = "oneYear_13",
                      varMulti=1.8,
                      breachesNum = 100,
                      seasonFreq = 93,
                      longThresholdMulti = 3, #
                      shortThreshold= 1.09,
                      tradingDays = 252,
                      BetaIndex = "OMXSPI_price")