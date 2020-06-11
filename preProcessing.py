import pandas as pd


def financial_ratios(rawData):
    """
    Compute multiple financial ratios
    Require quarterly or yearly financial data matrix as input
    :type rawData: data frame
    """

    # Profitability ratios
    operating_margin = rawData.operating_Income / rawData.revenues
    operating_profit_margin = operating_margin * 100
    net_income = rawData.earnings_Per_Share * rawData.number_Of_Shares
    return_on_total_equity = net_income / rawData.total_Equity
    return_on_total_liabilites_and_equity = net_income / rawData.total_Liabilities_And_Equity
    return_on_total_assets = net_income / rawData.total_Assets
    roa_DP = (net_income / rawData.revenues) * (rawData.revenues / rawData.total_Assets)
    gross_profit_margin = rawData.gross_Income / rawData.revenues
    cash_flow = rawData.cash_Flow_From_Operating_Activities + \
                rawData.cash_Flow_From_Investing_Activities + rawData.cash_Flow_From_Financing_Activities

    free_cash_flow = rawData.free_Cash_Flow
    capital_employed = rawData.total_Assets - rawData.current_Liabilities
    cash_flow_return_on_investment = rawData.cash_Flow_From_Operating_Activities / capital_employed
    Return_on_Capital_Employed = rawData.profit_Before_Tax / capital_employed
    net_gearing = rawData.net_Debt / rawData.total_Equity
    basic_earning_power_ratio = rawData.profit_Before_Tax / rawData.total_Assets
    return_on_net_assets = rawData.profit_Before_Tax / (
            rawData.tangible_Assets + rawData.current_Assets - rawData.current_Liabilities)

    # Liquidity ratios
    current_ratio = rawData.current_Assets / rawData.current_Liabilities
    quick_ratio = rawData.cash_And_Equivalents / rawData.current_Liabilities
    operating_cash_flow_ratio = rawData.cash_Flow_From_Operating_Activities / rawData.net_Debt

    # Activity ratios / efficiency ratios
    asset_turnover = rawData.revenues / rawData.total_Assets

    # Debt ratios / leveraging ratios
    debt_ratio = (rawData.current_Liabilities + rawData.non_Current_Liabilities) / rawData.total_Assets
    debt_to_equity_ratio = rawData.net_Debt / rawData.total_Equity
    total_liability_to_equity_ratio = (rawData.current_Liabilities + rawData.non_Current_Liabilities) / \
                                      rawData.total_Equity

    # Market ratios
    payout_ratio = rawData.dividend / rawData.earnings_Per_Share
    dividend_cover = rawData.earnings_Per_Share / rawData.dividend
    PE_ratio = rawData.stock_Price_Average / rawData.earnings_Per_Share
    total_shareholders_equity = rawData.total_Assets - (rawData.non_Current_Liabilities + rawData.current_Liabilities)
    book_value_per_share = total_shareholders_equity / rawData.number_Of_Shares
    PB_ratio = rawData.stock_Price_Average / book_value_per_share
    PS_ratio = rawData.stock_Price_Average / rawData.revenues
    market_capitalization = rawData.stock_Price_Average * rawData.number_Of_Shares
    enterprise_value = market_capitalization + rawData.net_Debt - rawData.cash_And_Equivalents
    EV_EBITDA_ratio = enterprise_value / rawData.profit_Before_Tax
    dividend_yield = rawData.dividend / rawData.stock_Price_Average

    # Not computed
    # risk_adjusted_return_on_capital = expectedReturn / economicCapital
    # average colletion period = accounts recievable / (annual credit sales / 365 days)
    # degree of operating leverage = percent change in net operating income / percent change in sales
    # DSO ratio = accounts reciavable / (total annual sales / 365 days )
    # average payment period = accounts payable /(annual credit purchase / 365 days)
    # stock turnover ratio = cost of goods sold /average inventory
    # recievables turnover ratio = net credit sales / average net recievables
    # inventory conversion ratio = 365 days / inventory turnover
    # Receivables conversion period = recievables / net salwes   * 365 days
    # Payables conversion period = accounts payables / purchases
    # time_interest_earned_ratio = net income / annual interest expense
    # debt_service_coverage_ratio = net operating income / total debt service
    # roa_DP = (net_income / net_sales)*(net_sales / total_assets)
    # ROC = EBIT*(1-tax_rate)/invested_Capital
    # roe_DP = (net_income/rawStockQuarterD.revenues)*(rawStockQuarterD.revenues/average_assets)*
    # (average_assets/average_equity)
    # efficiency_ratio=non_intereset_expense/revenue
    # PEG_ratio = PE_ratio / annaul eps growth
    # cash_flow_ratio = rawData.stock_Price_Average / present value of cash flow per share

    ratiolist = pd.concat([operating_margin, operating_profit_margin, net_income, return_on_total_equity,
                           return_on_total_liabilites_and_equity, return_on_total_assets, roa_DP, gross_profit_margin,
                           cash_flow, free_cash_flow, capital_employed, cash_flow_return_on_investment,
                           Return_on_Capital_Employed, net_gearing, basic_earning_power_ratio, return_on_net_assets,
                           current_ratio, quick_ratio, operating_cash_flow_ratio, asset_turnover, debt_ratio,
                           debt_to_equity_ratio, total_liability_to_equity_ratio, payout_ratio, dividend_cover,
                           PE_ratio, total_shareholders_equity, book_value_per_share, PB_ratio, PS_ratio,
                           market_capitalization, enterprise_value, EV_EBITDA_ratio, dividend_yield],
                          axis=1, sort=False,
                          keys=["operating_margin", "operating_profit_margin", "net_income", "return_on_total_equity",
                                "return_on_total_liabilites_and_equity", "return_on_total_assets", "roa_DP",
                                "gross_profit_margin", "cash_flow", "free_cash_flow", "capital_employed",
                                "cash_flow_return_on_investment", "Return_on_Capital_Employed", "net_gearing",
                                "basic_earning_power_ratio", "return_on_net_assets", "current_ratio", "quick_ratio",
                                "operating_cash_flow_ratio", "asset_turnover", "debt_ratio", "debt_to_equity_ratio",
                                "total_liability_to_equity_ratio", "payout_ratio", "dividend_cover", "PE_ratio",
                                "total_shareholders_equity", "book_value_per_share", "PB_ratio", "PS_ratio",
                                "market_capitalization", "enterprise_value", "EV_EBITDA_ratio", "dividend_yield"])

    return ratiolist


def prepare_quarter_data(quarter_data):
    # TODO extract the last chunk that i delete, make that the future test set, what I use for models
    """
    return: Lagged data frame of quarterly report with financial ratios
    """
    i = 0
    for stock in quarter_data:
        returns = quarter_data[stock]["stock_Price_Average"].pct_change(-1)
        # returns.drop(returns.tail(1).index, inplace=True)

        quarter_ratios = financial_ratios(quarter_data[stock])
        # quarter_ratios.drop(quarter_ratios.tail(1).index, inplace=True)
        quarter_values = quarter_data[stock].drop(["stock_Price_Average", "stock_Price_High", "stock_Price_Low"],axis=1)
        # quarter_values.drop(quarter_values.tail(1).index, inplace=True)

        returns.reset_index(drop=True, inplace=True)
        quarter_ratios.reset_index(drop=True, inplace=True)
        quarter_values.reset_index(drop=True, inplace=True)

        full_quarter_matrix = pd.concat([returns,
                                         quarter_ratios,
                                         quarter_values], axis=1)
        full_quarter_matrix["stock"] = stock
        full_quarter_matrix.columns = full_quarter_matrix.columns.str.replace('stock_Price_Average', 'returns')

        if i == 0:
            all_stocks_matrix = full_quarter_matrix
        else:
            all_stocks_matrix = pd.concat([all_stocks_matrix, full_quarter_matrix], axis=0)
        i=+1
    return all_stocks_matrix
