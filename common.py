def instrumentDictionary(jsonRequest):

    # Create list of tickers with additional data
    names_list = []
    urlName_list = []
    instrument_list = []
    ticker_list = []
    sectorId_list = []
    marketId_list = []
    countryId_list = []
    insId_list = []

    for tickIterator in jsonRequest["instruments"]:
        # tickIterator['name']
        temp1 = tickIterator['name']
        temp2 = tickIterator['urlName']
        temp3 = tickIterator['instrument']
        temp4 = tickIterator['ticker']
        temp5 = tickIterator['sectorId']
        temp6 = tickIterator['marketId']
        temp7 = tickIterator['countryId']
        temp8 = tickIterator['insId']
        names_list.append(temp1)
        urlName_list.append(temp2)
        instrument_list.append(temp3)
        ticker_list.append(temp4)
        sectorId_list.append(temp5)
        marketId_list.append(temp6)
        countryId_list.append(temp7)
        insId_list.append(temp8)

    return names_list, urlName_list, instrument_list, ticker_list, sectorId_list, marketId_list, countryId_list, \
           insId_list;



def idConv(tickerName, ticker_list, insId_list):
    indexTemp = ticker_list.index(tickerName)
    return str(insId_list[indexTemp])




def getCountry(id, dataCountries):
    for item in dataCountries:
        if item['id'] == id:
            break
    return item['name']



def getMarket(id, dataMarkets):
    for item in dataMarkets:
        if item['id'] == id:
            break
    return item['name']


def getSector(id, dataSectors):
    for item in dataSectors:
        if item['id'] == id:
            break
    return item['name']