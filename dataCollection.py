import requests
import os
from common import idConv
from common import instrumentDictionary
from common import getSector
from common import getMarket
from common import getCountry
from time import sleep
import numpy as np
import pandas as pd


class collectTickerData:
    # Initializer / Instance Attributes
    def __init__(self, tickerName, ticker_list, insId_list, apiKey):
        self.name = tickerName
        sleep(0.5)
        """Price data """
        tempPriceData = requests.get("https://apiservice.borsdata.se/v1/instruments/" + idConv(
            tickerName, ticker_list, insId_list) + "/stockprices?authKey=" + apiKey).json()
        tickerPriceData = []
        # print(tempPriceData["stockPricesList"])
        for item in tempPriceData["stockPricesList"]:
            tempPrice = [item["d"], item["c"], item["o"], item["h"], item["l"], item["v"]]
            tickerPriceData.append(tempPrice)
        tickerPriceDataFrame = pd.DataFrame(data=tickerPriceData,
                                            columns=("Time", "Close", "Open", "High", "Low", "volume"))
        self.priceData = tickerPriceDataFrame
        sleep(0.5)
        """ Year Data """
        tempYear = requests.get(
            "https://apiservice.borsdata.se/v1/instruments/" + idConv(tickerName, ticker_list, insId_list) +
            "/reports/year?authKey=" + apiKey).json()
        yearData = []

        # print(tempYear["reports"])
        try:
            for item in tempYear["reports"]:
                yearTemp = [item["report_End_Date"], item["revenues"], item["gross_Income"],
                            item["operating_Income"], item["profit_Before_Tax"],
                            item["profit_To_Equity_Holders"], item["earnings_Per_Share"],
                            item["number_Of_Shares"], item["dividend"], item["intangible_Assets"],
                            item["tangible_Assets"], item["financial_Assets"], item["non_Current_Assets"],
                            item["cash_And_Equivalents"], item["current_Assets"], item["total_Assets"],
                            item["total_Equity"], item["non_Current_Liabilities"],
                            item["current_Liabilities"], item["total_Liabilities_And_Equity"],
                            item["net_Debt"], item["cash_Flow_From_Operating_Activities"],
                            item["cash_Flow_From_Investing_Activities"],
                            item["cash_Flow_From_Financing_Activities"], item["cash_Flow_For_The_Year"],
                            item["free_Cash_Flow"], item["stock_Price_Average"], item["stock_Price_High"],
                            item["stock_Price_Low"]]
                yearData.append(yearTemp)
            yearDataFrame = pd.DataFrame(data=yearData,
                                         columns=("report_End_Date", "revenues", "gross_Income", "operating_Income",
                                                  "profit_Before_Tax", "profit_To_Equity_Holders",
                                                  "earnings_Per_Share",
                                                  "number_Of_Shares", "dividend", "intangible_Assets",
                                                  "tangible_Assets",
                                                  "financial_Assets", "non_Current_Assets", "cash_And_Equivalents",
                                                  "current_Assets", "total_Assets", "total_Equity",
                                                  "non_Current_Liabilities",
                                                  "current_Liabilities", "total_Liabilities_And_Equity", "net_Debt",
                                                  "cash_Flow_From_Operating_Activities",
                                                  "cash_Flow_From_Investing_Activities",
                                                  "cash_Flow_From_Financing_Activities", "cash_Flow_For_The_Year",
                                                  "free_Cash_Flow", "stock_Price_Average", "stock_Price_High",
                                                  "stock_Price_Low"))
            self.yearData = yearDataFrame
            # tempRr12 = requests.get("https://apiservice.borsdata.se/v1/instruments/" + idConv(tickerName) + "/reports/r12?authKey=" + apiKey).json()
            # tempRr12["reports"][0].keys()
            # tempRr12["reports"][0]["report_Start_Date"]
            # tempRr12["reports"][0]["report_End_Date"]
            print("Year data read")
        except Exception:
            print("couldn't read year data")
            print(tempYear)
        sleep(0.5)
        """Quarter data"""
        tempQuarter = requests.get(
            "https://apiservice.borsdata.se/v1/instruments/" + idConv(tickerName, ticker_list, insId_list) +
            "/reports/quarter?authKey=" + apiKey).json()
        quarterData = []
        # print(tempQuarter["reports"])
        try:
            for item in tempQuarter["reports"]:
                quarterTemp = [item["report_End_Date"], item["revenues"], item["gross_Income"],
                               item["operating_Income"], item["profit_Before_Tax"],
                               item["profit_To_Equity_Holders"], item["earnings_Per_Share"],
                               item["number_Of_Shares"], item["dividend"], item["intangible_Assets"],
                               item["tangible_Assets"], item["financial_Assets"], item["non_Current_Assets"],
                               item["cash_And_Equivalents"], item["current_Assets"], item["total_Assets"],
                               item["total_Equity"], item["non_Current_Liabilities"],
                               item["current_Liabilities"], item["total_Liabilities_And_Equity"],
                               item["net_Debt"], item["cash_Flow_From_Operating_Activities"],
                               item["cash_Flow_From_Investing_Activities"],
                               item["cash_Flow_From_Financing_Activities"], item["cash_Flow_For_The_Year"],
                               item["free_Cash_Flow"], item["stock_Price_Average"], item["stock_Price_High"],
                               item["stock_Price_Low"]]
                quarterData.append(quarterTemp)
            quarterDataFrame = pd.DataFrame(data=quarterData,
                                            columns=("report_End_Date", "revenues", "gross_Income", "operating_Income",
                                                     "profit_Before_Tax", "profit_To_Equity_Holders",
                                                     "earnings_Per_Share",
                                                     "number_Of_Shares", "dividend", "intangible_Assets",
                                                     "tangible_Assets",
                                                     "financial_Assets", "non_Current_Assets", "cash_And_Equivalents",
                                                     "current_Assets", "total_Assets", "total_Equity",
                                                     "non_Current_Liabilities",
                                                     "current_Liabilities", "total_Liabilities_And_Equity", "net_Debt",
                                                     "cash_Flow_From_Operating_Activities",
                                                     "cash_Flow_From_Investing_Activities",
                                                     "cash_Flow_From_Financing_Activities", "cash_Flow_For_The_Year",
                                                     "free_Cash_Flow", "stock_Price_Average", "stock_Price_High",
                                                     "stock_Price_Low"))
            print("Quarter data read")
            self.quarterData = quarterDataFrame
        except Exception:
            print("couldn't read quarter data")
            print(tempQuarter)


def read_files_from_disk(tickerList,
                         selectedTickers=["ATRE", "AZN", "BILL"],
                         selectedSectors="all",
                         selectedCountries="all",
                         selectedMarket="all"):  # Ticker, Sector or Market
    readDatasets = {}
    if selectedMarket == "all" and selectedCountries == "all" and selectedSectors == "all" and selectedTickers == "all":
        for ticker in tickerList["Ticker"]:
            year = pd.read_csv(
                filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/data/" + ticker + "/" + ticker + "Year" + ".csv")
            quarter = pd.read_csv(
                filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/data/" + ticker + "/" + ticker + "Quarter" + ".csv")
            price = pd.read_csv(
                filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/data/" + ticker + "/" + ticker + "price" + ".csv")

            readDatasets[ticker + "_year"] = year
            readDatasets[ticker + "_quarter"] = quarter
            readDatasets[ticker + "_price"] = price
    else:
        indecesTickers = np.array(None)
        indecesSectors = np.array(None)
        indecesCountries = np.array(None)
        indecesMarkets = np.array(None)

        if selectedTickers != "all":
            bool = tickerList['Ticker'].isin(selectedTickers)
            indecesTickers = np.where(bool == True)

        if selectedSectors != "all":
            bool = tickerList['Sector'].isin(selectedSectors)
            indecesSectors = np.where(bool == True)

        if selectedCountries != "all":
            bool = tickerList['Country'].isin(selectedCountries)
            indecesCountries = np.where(bool == True)

        if selectedMarket != "all":
            bool = tickerList['Market'].isin(selectedMarket)
            indecesMarkets = np.where(bool == True)

        # unionCond = (selectedMarket != "all" , selectedCountries == "all" , selectedSectors == "all" , selectedTickers == "all" )
        # unionCond = np.where(np.array(unionCond)==True)
        # if(len(unionCond) >= 2):
        # indexList = list(np.concatenate((indecesTickers, indecesSectors, indecesCountries, indecesMarkets), axis=None)     )
        # indexList = list(indexList)
        # l = [1,2,3,4,4,5,5,6,1]
        # set([x for x in indexList if indexList.count(x) > 1])

        # set(indexList[0]).intersection(*indexList[1:])

        keptIndices = np.concatenate((indecesTickers, indecesSectors, indecesCountries, indecesMarkets), axis=None)
        keptIndices = keptIndices[keptIndices != np.array(None)]
        keptIndices = np.unique(keptIndices)
        tickerSubsetSelected = tickerList.iloc[keptIndices]

        for ticker in tickerSubsetSelected["Ticker"]:
            year = pd.read_csv(
                filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/data/" + ticker + "/" + ticker + "Year" + ".csv")
            quarter = pd.read_csv(
                filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/data/" + ticker + "/" + ticker + "Quarter" + ".csv")
            price = pd.read_csv(
                filepath_or_buffer="/Users/august/PycharmProjects/-BorsData-master/data/" + ticker + "/" + ticker + "price" + ".csv")
            readDatasets[ticker + "_year"] = year
            readDatasets[ticker + "_quarter"] = quarter
            readDatasets[ticker + "_price"] = price
    return readDatasets


def collect_ticker_metadata(apiKey, currentWD):
    # collectTickerData(apiKey=apiKey):
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
    insId_list = instrumentDictionary(dataInstruments)  # Fetch all ticker data

    sectorId_list_transl = []
    for value in sectorId_list:
        temp = getSector(value, dataSectors['sectors'])
        sectorId_list_transl.append(temp)

    marketId_list_transl = []
    for value in marketId_list:
        temp = getMarket(value, dataMarkets['markets'])
        marketId_list_transl.append(temp)

    countryId_list_transl = []
    for value in countryId_list:
        temp = getCountry(value, dataCountries['countries'])
        countryId_list_transl.append(temp)

    try:
        os.mkdir(currentWD + "/data")
    except OSError:
        print("Folder already exists")

    pd.DataFrame(
        {'Name': names_list, 'Ticker': ticker_list, 'Sector': sectorId_list_transl, 'Market': marketId_list_transl,
         'Country': countryId_list_transl}).to_csv(path_or_buf=currentWD + "/data/Tickers.csv", sep=",",
                                                   index=False, decimal=".", encoding="utf-8")
    return names_list, urlName_list, instrument_list, ticker_list, sectorId_list, marketId_list, countryId_list, insId_list


def download_all_data(tickerReadSet, ticker_list, insId_list, apiKey, currentWD):
    tickerObjectList = []
    i = 0
    # TODO make sure todays date is the latest, it is the same
    for item in tickerReadSet['Ticker']:
        i = i + 1
        print('Reading item', i, ",", item)
        try:
            tempObject = collectTickerData(item, ticker_list=ticker_list, insId_list=insId_list, apiKey=apiKey)
            tickerObjectList.append(tempObject)
        except Exception:
            print(item, 'failed collection')
        try:
            os.mkdir(currentWD + "/data/" + item)
            print(item, 'folder created')
        except Exception:
            print(item, 'folder already exists')
        try:
            tempObject.priceData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "price.csv", sep=",",
                                        header=True, index=False, decimal=".")
            print(item, 'price data created')
        except Exception:
            print(item, "couldn't write price data")
        try:
            tempObject.quarterData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "Quarter.csv", sep=",",
                                          header=True, index=False, decimal=".")
            print(item, 'quarter data created')
        except Exception:
            print(item, "couldn't write quarter data")
        try:
            tempObject.yearData.to_csv(path_or_buf=currentWD + "/data/" + item + "/" + item + "Year.csv", sep=",",
                                       header=True, index=False, decimal=".")
            print(item, 'year data created')
        except Exception:
            print(item, "couldn't write year data")
