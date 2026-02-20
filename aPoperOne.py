import pandas as pd
import sklearn
import yfinance
import numpy as np

def snp500pred(stocks, number):
    
    raw_data = yfinance.download(stocks, start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
    daily_difference_of_logs = (raw_data.shift(-1).apply(np.log) - raw_data.shift(1).apply(np.log))[1:-1]
    
    snp = yfinance.download('^GSPC', start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
    daily_difference_of_logs_snp = (snp.shift(-1).apply(np.log) - snp.shift(1).apply(np.log))[1:-1]
    
    best_explanatory = []
    explanatory_betas = []

    for n in range(number):
        print(n)
        print(best_explanatory)
        explanatory_stocks = daily_difference_of_logs.get(best_explanatory)
        remaining_stocks = daily_difference_of_logs.copy()
        if (n != 0):
            remaining_stocks = remaining_stocks.drop(best_explanatory, axis = 1, inplace = False)

        best_r2 = -1
        best_beta = 0
        best_stock = ""

        for stock in remaining_stocks.columns:
            
            stock_data = remaining_stocks.get(stock)

            reg = sklearn.linear_model.LinearRegression()
            reg.fit(pd.concat([explanatory_stocks, stock_data], axis = 1), daily_difference_of_logs_snp)
            
            r2 = sklearn.metrics.r2_score(daily_difference_of_logs_snp, reg.predict(pd.concat([explanatory_stocks, stock_data], axis = 1)))

            if r2 > best_r2:
                best_r2 = r2
                best_beta = reg.coef_
                best_stock = stock
        print(best_beta)
        print(best_stock)
        best_explanatory.append(best_stock)
        explanatory_betas.append(best_beta)

    return [best_explanatory, explanatory_betas]