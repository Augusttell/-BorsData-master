import numpy as np
import pandas as pd
import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import norm
import scipy


def valuation():
    # TODO Implement classical valuation methods ?


def momentum(subsetDay: str,
                      lengthOfData: int = 1095,
                      seasonalityLength: int = 93,
                      microLengthRegression: int = 62,
                      repo: str = "_",
                      prefix: str = "",
                      seasonFreq: int = 93,
                      longThresholdMulti: int = 3, #
                      shortThreshold: float = 1.02,
                      varMulti:float = 1.0,
                      breachesNum:int = 1,
                      tradingDays:int = 252,
                      BetaIndex: str = "OMXSPI_price") -> str:
    stockList = []
    timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')

    betaIndex = subsetDay[BetaIndex][-(lengthOfData + seasonalityLength):]

    resultIndex = seasonal_decompose(betaIndex.Close,
                                model='multiplicative',  # multiplicative, additive
                                freq=seasonFreq,
                                two_sided=False)

    naLengthInd = resultIndex.trend.isna().sum()


    regressIndex = scipy.stats.linregress(x=np.arange(naLengthInd, betaIndex.shape[0]), y=resultIndex.trend[naLengthInd:])
    returnsSetIndex = betaIndex["Close"].pct_change(1)
    indexVar = (returnsSetIndex.var() / tradingDays)/returnsSetIndex.mean()

    betaIndex.reset_index(inplace=True)
    betaIndex['Time'] = pd.to_datetime(betaIndex['Time'])

    try:
        os.mkdir(os.getcwd() + "/" + repo + "/" + "momentum_" + prefix + "_" + timeVar + "/")
        print('folder created')
    except OSError:
        print('folder already exists')
    for stock in subsetDay:
        print(stock)
        priceSet = subsetDay[stock][-(lengthOfData + seasonalityLength):]
        if (priceSet['Time'].shape[0] < lengthOfData + seasonalityLength):
            print("too few observations!")
            continue

        else:

            returnsSet = priceSet["Close"].pct_change(1)

            tradingDays = 252
            tradeVar = (returnsSet.var() / tradingDays)/returnsSet.mean()
            if tradeVar> indexVar*varMulti:
                print("Greater volatility than index")
                continue

            try:
                result = seasonal_decompose(priceSet.Close,
                                            model='multiplicative',  # multiplicative, additive
                                            freq=seasonFreq,
                                            two_sided=False)

                # seasonalEffect = result.seasonal
                trendEffect = result.trend

            except OSError:
                print(stock + " failed")
                continue

            naLength = trendEffect.isna().sum()
            regressMeanLong = scipy.stats.linregress(x=np.arange(naLength, priceSet.shape[0]),
                                                      y=trendEffect[naLength:])

            breaches = np.where((np.arange(0, betaIndex.shape[0]) * regressIndex.slope + regressMeanLong.intercept - priceSet.Close) > 0)
            breaches = breaches[0]

            if breaches.size >= breachesNum:
                print("too many breaches!")
                continue

            if regressMeanLong.slope <= 0:
                print("Negative trend")
                continue

            if regressMeanLong.slope < regressIndex.slope:
                print("Less growth than index")
                continue

            if regressMeanLong.slope > regressIndex.slope*longThresholdMulti:
                print(str(longThresholdMulti) + " times larger than index!")
                continue

            regressMeanShort = scipy.stats.linregress(x=np.arange((priceSet.shape[0] - microLengthRegression),
                                                                  priceSet.shape[0]), y=trendEffect[-microLengthRegression:])
            if regressMeanShort.slope > shortThreshold:
                print("Overbought")
                continue

            priceSet.reset_index(inplace=True)
            priceSet['Time'] = pd.to_datetime(priceSet['Time'])


            plotMade = sns.lineplot(x='Time', y='Close',
                                    data=priceSet, c = "green")

            plotMade.plot(priceSet["Time"],
                          np.arange(0, priceSet.shape[0]) * regressMeanLong.slope + regressMeanLong.intercept,
                          ls='-', c="dodgerblue", label="Regression line")

            plotMade.plot(priceSet["Time"][-microLengthRegression:],
                          np.arange((priceSet.shape[0] - microLengthRegression),
                                    priceSet.shape[0]) * regressMeanShort.slope + regressMeanShort.intercept,
                              ls='-', c="mediumblue", label="Micro regression line")

            plotMade.plot(priceSet["Time"],
                          np.arange(0, betaIndex.shape[0]) * regressIndex.slope + regressMeanLong.intercept, ls='-',
                          c="red", label="Regression line Index ")

            plt.xticks(rotation=90)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title(stock[:-6])
            plotMade.figure.savefig(
                os.getcwd() + "/" + repo + "/" + "momentum_" + prefix + "_" + timeVar + "/" + stock[:-6] + ".png",
                bbox_inches='tight')  # save the figure to file
            plt.close(plotMade.figure)
            tempstock = stock[:-6]
            stockList.append(tempstock)
        df = pd.DataFrame({'sparade': stockList})
        df.to_csv(
            path_or_buf=os.getcwd() + "/" + repo + "/" + "momentum_" + prefix + "_" + timeVar + "/" + "intressanta.csv",
            sep=",",
            header=True, index=False, decimal=".")
        stringOfFile = os.getcwd() + "/" + repo + "/" + "momentum_" + prefix + "_" + timeVar + "/" + "intressanta.csv"

    return stringOfFile





def volatilityBreaker(subsetDay: str, lengthOfData: int = 365, seasonalityLength: int = 31,
                      augustCoefficient: int = 2, repo: str = "_",
                      prefix: str = "",
                      breachesNum: int = 50,
                      breachMethod: object = "lower", shiftThreshold: int = 3, seasonFreq: int = 62) -> str:
    stockList = []
    timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')
    try:
        os.mkdir(os.getcwd() + "/" + repo + "/" + "volatility_" + prefix + "_" + timeVar + "/")
        print('folder created')
    except OSError:
        print('folder already exists')
    for stock in subsetDay:
        print(stock)
        priceSet = subsetDay[stock][-(lengthOfData + seasonalityLength):]
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
                stockVar = np.sum(np.power((priceSet.Close - trendEffect), 2)) * (
                        1 / priceSet.Close.shape[0]) * augustCoefficient

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

            if (any(df.meanNegVar < 0)):
                continue

            if breachMethod == "upper":
                breaches = upperLimitBreach[0]
            if breachMethod == "lower":
                breaches = lowerLimitBreach[0]

            if breaches.size >= breachesNum or breaches.size == 0:
                if breaches.size >= breachesNum:
                    print("too many breaches!")
                    continue
                else:
                    print("Too small volatility")
                    continue
            else:
                regressMean = scipy.stats.linregress(x=np.arange(0, df.shape[0]), y=df.close[0:])

                if prefix == "twoWeek" or prefix == "oneWeek":
                    regressMeanShort = scipy.stats.linregress(x=np.arange((df.shape[0] - 62), df.shape[0]),
                                                              y=df.close[-62:])

                    if regressMeanShort.slope > 0.9:
                        print("To steep micro-trend! ")
                        continue

                if (regressMean.slope < 0):
                    print("Negative trend! ")
                    continue

                else:
                    boolSeq = np.arange(0, df.shape[0]) * regressMean.slope + regressMean.intercept < df.meanValue
                    shift = 0
                    oldState = boolSeq.iloc[0]
                    for val in boolSeq:
                        if (val == True):
                            newState = True
                            if (oldState == False):
                                shift += 1
                                oldState = True
                        if (val == False):
                            newState = False
                            if (oldState == True):
                                shift += 1
                                oldState = False
                    if shift <= shiftThreshold:
                        print("Too few shifts")
                        continue
                    else:
                        plt.figure(figsize=(14, 8))

                        if prefix == "twoWeek" or prefix == "oneWeek":
                            plotLength=df.shape[0]-180
                        else:
                            plotLength=df.shape[0]


                        plotMade = sns.lineplot(x='time', y='value', hue='variable',
                                                data=pd.melt(df[-plotLength:], ['time']))

                        plotMade.plot(df["time"].iloc[lowerLimitBreach[0][lowerLimitBreach[0]>=plotLength]],
                                      df["close"].iloc[lowerLimitBreach[0][lowerLimitBreach[0]>=plotLength]], 'g*',
                                      c="purple")

                        plotMade.plot(df["time"].iloc[lowerLimitBreach[0][lowerLimitBreach[0]>=plotLength]],
                                      df["close"].iloc[lowerLimitBreach[0][lowerLimitBreach[0]>=plotLength]], 'g*',
                                      c="purple")

                        plotMade.plot(df["time"][-plotLength:], np.arange(df.shape[0]-plotLength, df.shape[0]) * regressMean.slope + regressMean.intercept,
                                      ls='-', c="black", label="Regression line")
                        if prefix == "twoWeek" or prefix == "oneWeek":
                            plotMade.plot(df["time"][-62:], np.arange((df.shape[0] - 62), df.shape[0]) * regressMeanShort.slope +
                                          regressMeanShort.intercept,
                                          ls='-', c="black", label="Micro regression line")

                        plt.xticks(rotation=90)
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.title(stock[:-6])
                        plotMade.figure.savefig(
                            os.getcwd() + "/" + repo + "/" + "volatility_" + prefix + "_" + timeVar + "/" + stock[
                                                                                                            :-6] + ".png",
                            bbox_inches='tight')  # save the figure to file
                        # plotMade.plt.close()
                        plt.close(plotMade.figure)
                        tempstock = stock[:-6]
                        stockList.append(tempstock)
    df = pd.DataFrame({'sparade': stockList})
    df.to_csv(
        path_or_buf=os.getcwd() + "/" + repo + "/" + "volatility_" + prefix + "_" + timeVar + "/" + "intressanta.csv",
        sep=",",
        header=True, index=False, decimal=".")
    stringOfFile = os.getcwd() + "/" + repo + "/" + "volatility_" + prefix + "_" + timeVar + "/" + "intressanta.csv"
    return stringOfFile


def stateSpaceModel(subsetDay: str, tickersUnderStudy: str, foreCastLength: int, lengthOfData: int = 365,
                    seasonalityLength: int = 31,
                    repo: str = "_", includeCycle: bool = False,
                    prefix: str = "", threshold: object = 1.1) -> str:
    buyPrice = []
    sellPrice = []
    decision = []
    stockList = []
    percGain = []
    timeVar = datetime.datetime.now().strftime('%Y_%m_%d_%H')
    try:
        os.mkdir(os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/")
        print('folder created')
    except OSError:
        print('folder already exists')

    for stock in subsetDay:
        temp1 = 0
        temp2 = 0
        decisionTemp = "No"
        tickerLogic = tickersUnderStudy != stock[:-6]
        if (np.all(tickerLogic)):
            continue
        else:

            try:
                os.mkdir(
                    os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock + "/")
                print('folder created for ' + stock)
            except OSError:
                print('folder for stock ' + stock + ' already exists')

            priceSet = subsetDay[stock][-(lengthOfData + seasonalityLength):]
            if includeCycle == True:
                mod_cycle = sm.tsa.UnobservedComponents(priceSet["Close"],
                                                        level='lldtrend',  # strend, llevel, lldtrend
                                                        trend=True,
                                                        seasonal=seasonalityLength,
                                                        stochastic_level=True,
                                                        irregular=True,
                                                        cycle=includeCycle,
                                                        cycle_period_bounds=(14, 100))
            else:
                mod_cycle = sm.tsa.UnobservedComponents(priceSet["Close"],
                                                        level='lldtrend',  # strend, llevel, lldtrend
                                                        trend=True,
                                                        seasonal=seasonalityLength,
                                                        stochastic_level=True,
                                                        irregular=True)

            res_cycle = mod_cycle.fit()


            # observed vs predicted
            predictionsData = res_cycle.filter_results.forecasts[0]
            std_errors = np.sqrt(res_cycle.filter_results.forecasts_error_cov[0, 0])
            alpha = 0.05
            critical_value = norm.ppf(1 - alpha / 2.)
            ci_lower = predictionsData - critical_value * std_errors
            ci_upper = predictionsData + critical_value * std_errors

            data = {'Time': priceSet["Time"], 'observed': priceSet["Close"],
                    'predicted': predictionsData,
                    'lower': ci_lower, 'upper': ci_upper}

            df = pd.DataFrame(data)
            df.reset_index(inplace=True)
            df['Time'] = pd.to_datetime(df['Time'])
            plt.figure(figsize=(14, 8))
            plotMade = sns.lineplot(x='Time', y='observed', data=df[-300:], label='Observed')
            plotMade.plot(df.Time[-300:], df.predicted[-300:], color='k',
                          label='Predicted')
            plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title("Predicted vs observed " + stock[:-6])
            plt.xticks(rotation=90)
            plotMade.figure.savefig(
                os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
                "/predVsObs.png", bbox_inches='tight')
            plt.close(plotMade.figure)

            # level
            levelData = res_cycle.level["smoothed"]
            std_errors = np.sqrt(res_cycle.level["smoothed_cov"])
            ci_lower = levelData - critical_value * std_errors
            ci_upper = levelData + critical_value * std_errors

            data = {'Time': priceSet["Time"], 'Level': levelData,
                    'lower': ci_lower, 'upper': ci_upper}

            df = pd.DataFrame(data)
            df.reset_index(inplace=True)
            df['Time'] = pd.to_datetime(df['Time'])
            plt.figure(figsize=(14, 8))
            plotMade = sns.lineplot(x='Time', y='Level', data=df[-300:], label='Level')

            plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title("Level " + stock[:-6])
            plt.xticks(rotation=90)
            plotMade.figure.savefig(
                os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
                "/level.png", bbox_inches='tight')
            plt.close(plotMade.figure)

            # Trend
            trendData = res_cycle.trend["smoothed"]
            std_errors = np.sqrt(res_cycle.trend["smoothed_cov"])
            ci_lower = trendData - critical_value * std_errors
            ci_upper = trendData + critical_value * std_errors

            data = {'Time': priceSet["Time"], 'Trend': trendData,
                    'lower': ci_lower, 'upper': ci_upper}

            df = pd.DataFrame(data)
            df.reset_index(inplace=True)
            df['Time'] = pd.to_datetime(df['Time'])
            plt.figure(figsize=(14, 8))
            plotMade = sns.lineplot(x='Time', y='Trend', data=df[-300:], label='Trend')

            plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title("Trend " + stock[:-6])
            plt.xticks(rotation=90)
            plotMade.figure.savefig(
                os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
                "/trend.png", bbox_inches='tight')
            plt.close(plotMade.figure)

            if includeCycle == True:
                cycleData = res_cycle.cycle["smoothed"]
                std_errors = np.sqrt(res_cycle.cycle["smoothed_cov"])
                ci_lower = cycleData - critical_value * std_errors
                ci_upper = cycleData + critical_value * std_errors

                data = {'Time': priceSet["Time"], 'Cycle': cycleData,
                        'lower': ci_lower, 'upper': ci_upper}

                df = pd.DataFrame(data)
                df.reset_index(inplace=True)
                df['Time'] = pd.to_datetime(df['Time'])
                plt.figure(figsize=(14, 8))
                plotMade = sns.lineplot(x='Time', y='Cycle', data=df[-300:], label='Cycle')

                plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.title("Cycle " + stock[:-6])
                plt.xticks(rotation=90)
                plotMade.figure.savefig(
                    os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
                    "/Cycle.png", bbox_inches='tight')
                plt.close(plotMade.figure)

            # Seasonal
            seasonalData = res_cycle.seasonal["smoothed"]
            std_errors = np.sqrt(res_cycle.seasonal["smoothed_cov"])
            ci_lower = seasonalData - critical_value * std_errors
            ci_upper = seasonalData + critical_value * std_errors

            data = {'Time': priceSet["Time"], 'Seasonal': seasonalData,
                    'lower': ci_lower, 'upper': ci_upper}

            df = pd.DataFrame(data)
            df.reset_index(inplace=True)
            df['Time'] = pd.to_datetime(df['Time'])
            plt.figure(figsize=(14, 8))
            plotMade = sns.lineplot(x='Time', y='Seasonal', data=df[-300:], label='Seasonal')

            plt.fill_between(df.Time[-300:], df.lower[-300:], df.upper[-300:], alpha=0.2)
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title("Seasonal " + stock[:-6])
            plt.xticks(rotation=90)
            plotMade.figure.savefig(
                os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
                "/seasonal.png", bbox_inches='tight')
            plt.close(plotMade.figure)

            # Forecast
            if prefix == "twoWeek" or prefix == "oneWeek":
                plotLength = df.shape[0] - 180
            else:
                plotLength = df.shape[0]

            predict_df = res_cycle.forecast(foreCastLength)

            priceSet['Time'] = pd.to_datetime(priceSet['Time'])
            temp = priceSet["Time"].iloc[-1]
            t = datetime.timedelta(days=1)
            foreCastStartDate = temp + t
            predict_df.index = pd.date_range(start=foreCastStartDate, periods=foreCastLength, freq='D')
            plotMade = sns.lineplot(
                y=priceSet["Close"][-plotLength:], x=priceSet["Time"][-plotLength:])
            plotMade.plot(predict_df, c="purple")

            plt.xticks(rotation=90)
            plotMade.figure.savefig(
                os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + stock +
                "/prediction.png", bbox_inches='tight')
            plt.close(plotMade.figure)

            meanObserved = priceSet["Close"].iloc[-1]
            # meanObserved = priceSet["Close"][-1:].mean()

            meanForecast = round(np.mean(predict_df[-int(np.floor(foreCastLength / 3)):]), 2)

            pd.DataFrame(priceSet[-100:]).to_csv(
                path_or_buf=os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" +
                            timeVar + "/" + stock + "/Prices.csv", sep=",", header=True, index=False,
                decimal=".")

            pd.DataFrame(predict_df).to_csv(
                path_or_buf=os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" +
                            timeVar + "/" + stock + "/prediction.csv", sep=",", header=True, index=False,
                decimal=".")

            temp1 = meanObserved
            temp2 = meanForecast
            temp3 = meanForecast / meanObserved

            buyPrice.append(temp1)
            sellPrice.append(temp2)
            percGain.append(temp3)

            decisionTemp = "No"
            if (temp3 >= threshold):
                decisionTemp = "Yes"

            decision.append(decisionTemp)
            stockList.append(stock[:-6])
    df = pd.DataFrame(
        {'Aktie': stockList, "Köp pris": buyPrice, "Sälj pris": sellPrice, "Beslut": decision, "Procent": percGain})
    print(df)
    df.to_csv(
        path_or_buf=os.getcwd() + "/" + repo + "/" + "stateSpaceModelling_" + prefix + "_" + timeVar + "/" + "/aktier.csv",
        sep=",", header=True, index=False, decimal=".")


# https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f


# TODO https://www.statsmodels.org/stable/examples/index.html#statespace
# Beta/correlation computations
## https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9


# def correlationCompute(tickerList = c, whichMetric = , returnBeta = , )
#         '''' Computes correlation table between stocks
#             Beta_1 Is beta related to all selected stocks
#             Beta_2 is related to all stocks available
#             ''''
#
#     dataLink = "C://Users//Augus//PycharmProjects//BorsData//data"
#
#     import os
#     tickerReadSet[]
#
#     dirs = os.listdir(dataLink)
#
#     dirsList = []
#     for item in dirs:
#         temp = dataLink + "//" + item
#         dirsList.append(temp)
#
#     dataSet1 = pd.read_csv("C://Users//Augus//PycharmProjects//BorsData//data//AF B//AF BQuarter.csv")  # doctest: +SKIP
#     dataSet2 = pd.read_csv("C://Users//Augus//PycharmProjects//BorsData//data//ACRI//ACRIQuarter.csv")  # doctest: +SKIP
#
#
#     dataSet1['datetime'] = pd.to_datetime(dataSet1['report_End_Date']) # Convert time variable to time type
#     dataSet1 = dataSet1.set_index('datetime') # Set to time , make series
#     dataSet1.drop(['report_End_Date'], axis=1, inplace=True) # Drop time column
#
#     dataSet2['datetime'] = pd.to_datetime(dataSet2['report_End_Date']) # Convert time variable to time type
#     dataSet2 = dataSet2.set_index('datetime') # Set to time , make series
#     dataSet2.drop(['report_End_Date'], axis=1, inplace=True) # Drop time column
#
#
#
#     # compute returns
#     dataSet1['stock_Price_Average_avgReturn'] = dataSet1['stock_Price_Average'].pct_change(periods=-1)
#     dataSet2['stock_Price_Average_avgReturn'] = dataSet2['stock_Price_Average'].pct_change(periods=-1)
#
#     returnsList = []
#     # Correaltion between two
#     dataSet1["cash_Flow_From_Investing_Activities"].corr(dataSet2["cash_Flow_From_Investing_Activities"])
# result = pd.concat([dataSet1, dataSet2], axis=1)
# test.iloc[:1]
# # Need to choose variables earlier


# Randomly generate weights
def weightGeneration(numStocks,  # 1 * Possiblestocks
                     numPortfolios  # Possible permutations
                     ):
    numWeights = numStocks
    results = np.zeros((numPortfolios, numWeights))
    # Generate proposals:
    for i in range(numPortfolios):
        # select random weights for portfolio holdings
        weights = np.random.random(numWeights)
        # rebalance weights to sum to 1
        weights /= np.sum(weights)
        # weights=weights*100
        results[i,] = weights
    return results


def filterSelector(data, lengthPrice=200, lengthQuarter=16, lengthYear=4):
    subsetYear = {k: data[k] for k in [s for s in tickerKeys if "year" in s] if k in data}
    subsetQuarter = {k: data[k] for k in [s for s in tickerKeys if "quarter" in s] if k in data}
    subsetDay = {k: data[k] for k in [s for s in tickerKeys if "price" in s] if k in data}

    for key, item in subsetDay.items():
        subsetDay[key] = item[:lengthPrice]

    for key, item in subsetQuarter.items():
        subsetQuarter[key] = item[:lengthQuarter]

    for key, item in subsetYear.items():
        subsetYear[key] = item[:lengthYear]

    return subsetYear, subsetQuarter, subsetDay


def computeMoments(yearData, quarterData, dayData):
    # Year
    yeardDataDict = {}
    for stock in yearData:
        columns = list(yearData[stock].columns)
        meanYear = yearData[stock][columns[1:]].mean(axis=0)
        medianYear = yearData[stock][columns[1:]].median(axis=0)
        stdYear = yearData[stock][columns[1:]].std(axis=0)
        varYear = yearData[stock][columns[1:]].std(axis=0)
        meanGrowthYear = yearData[stock][columns[1:]].pct_change(periods=-1).mean(axis=0)
        stdGrowthYear = yearData[stock][columns[1:]].pct_change(periods=-1).std(axis=0)
        lastGrowthYear = yearData[stock][columns[1:]].pct_change(periods=-1)[:1]

        yearDataProcessed = pd.concat([meanYear, medianYear, stdYear, varYear, meanGrowthYear, stdGrowthYear,
                                       lastGrowthYear.transpose()], axis=1)
        yearDataProcessed.columns = ("meanYear", "medianYear", "stdYear", "varYear", "meanGrowthYear", "stdGrowthYear",
                                     "lastGrowthYear")
        yeardDataDict[stock] = yearDataProcessed

    # Quarter
    quarterDataDict = {}
    for stock in quarterData:
        columns = list(quarterData[stock].columns)
        meanYear = quarterData[stock][columns[1:]].mean(axis=0)
        medianYear = quarterData[stock][columns[1:]].median(axis=0)
        stdYear = quarterData[stock][columns[1:]].std(axis=0)
        varYear = quarterData[stock][columns[1:]].std(axis=0)
        meanGrowthYear = quarterData[stock][columns[1:]].pct_change(periods=-1).mean(axis=0)
        stdGrowthYear = quarterData[stock][columns[1:]].pct_change(periods=-1).std(axis=0)
        lastGrowthYear = quarterData[stock][columns[1:]].pct_change(periods=-1)[:1]

        quarterDataProcessed = pd.concat([meanYear, medianYear, stdYear, varYear, meanGrowthYear, stdGrowthYear,
                                          lastGrowthYear.transpose()], axis=1)
        quarterDataProcessed.columns = (
            "meanYear", "medianYear", "stdYear", "varYear", "meanGrowthYear", "stdGrowthYear",
            "lastGrowthYear")
        quarterDataDict[stock] = quarterDataProcessed

    # Price
    dayDataDict = {}
    for stock in dayData:
        columns = list(dayData[stock].columns)
        meanYear = dayData[stock][columns[1:]].mean(axis=0)
        medianYear = dayData[stock][columns[1:]].median(axis=0)
        stdYear = dayData[stock][columns[1:]].std(axis=0)
        varYear = dayData[stock][columns[1:]].std(axis=0)
        meanGrowthYear = dayData[stock][columns[1:]].pct_change(periods=-1).mean(axis=0)
        stdGrowthYear = dayData[stock][columns[1:]].pct_change(periods=-1).std(axis=0)

        dayDataProcessed = pd.concat([meanYear, medianYear, stdYear, varYear, meanGrowthYear, stdGrowthYear], axis=1)
        dayDataProcessed.columns = ("meanYear", "medianYear", "stdYear", "varYear", "meanGrowthYear", "stdGrowthYear")
        dayDataDict[stock] = dayDataProcessed

    return yeardDataDict, quarterDataDict, dayDataDict

# Ydata, Qdata, Ddata = filterSelector(test, lengthPrice=200, lengthQuarter=16, lengthYear=4)
# yeardDataDict, quarterDataDict, dayDataDict = computeMoments(yearData=Ydata, quarterData=Qdata, dayData=Ddata)
# weights = weightGeneration(3, 2000)
#
#
# portfolios = {}
#
# yeardDataDict["BILL_year"] * weights


# TODO make above function run in one loop

# TODO Add covariance


#     # calculate portfolio return and volatility
#     portfolio_return = np.sum(stock_mean_daily_returns * weights) * 252
#     portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
#
#     # convert results array to Pandas DataFrame
#     results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe'])
#
#
#
#
#
#     cov_matrix = returns.cov()
#
#     inputConstraintMatrix1,  # 1 * Possiblestocks
#     inputConstraintMatrix2,  # 1 * Possiblestocks
#     inputConstraintMatrix3,  # 1 * Possiblestocks
#
#
#
#     # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
#     results[2, i] = results[0, i] / results[1, i]
#
#     # calculate portfolio return and volatility
#     portfolio_return = np.sum(stock_mean_daily_returns * weights) * 252
#     portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
#
#     # convert results array to Pandas DataFrame
#     results_frame = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe'])
#
#     # create scatter plot coloured by Sharpe Ratio
#     plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
#     plt.colorbar()
#
#
# def optimalPortfolios():


# Maximization

# How is goal function defined?




# def train_model(quarter_data_frame):
#     def __init__(self, quarter_data):
#         asd
#     # TODO Add cross validation
#     # Take three quarters predict fourth
#     # TODO Add Hyperparameter optimization
#     # TODO Out fold predictive results
#     # TODO Add PDP
#     # TODO Add test set predictions ( future returns )
#
#     ''' Trains an xgboost model'''
#
#     # Predict outcome
#     # Automatic xgboost model
#     X_train, X_test, y_train, y_test = \
#         train_test_split(quarter_data_frame.loc[:, ~quarter_data_frame.columns.isin(['returns', 'report_End_Date', 'stock'])],
#                          quarter_data_frame["returns"], test_size=0.2, random_state=42)
#
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtest = xgb.DMatrix(X_test, label=y_test)
#     param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
#     param['nthread'] = 4
#     param['eval_metric'] = 'rmse'
#
#     num_round = 10
#     evallist = [(dtest, 'eval'), (dtrain, 'train')]
#
#     bst = xgb.train(param, dtrain, num_round, evallist)
#     ypred = bst.predict(dtest)
#     xgb.plot_importance(bst)
#     return ypred
