import pandas as pd
import csv
import os
import sklearn
import yfinance
import numpy as np

# data = np.genfromtxt('snpNumbers.txt', delimiter = ',')

# first = data[0]
# second = data[1]

# firstLogDifference = np.log(first[1:]) - np.log(first[:-1])
# secondLogDifference = np.log(second[1:]) - np.log(second[:-1])


# reg = sklearn.linear_model.LinearRegression()
# reg.fit(firstLogDifference.reshape(-1, 1), secondLogDifference)
# print(reg.coef_)
# print(reg.intercept_)
# print(sklearn.metrics.r2_score(secondLogDifference, reg.predict(firstLogDifference.reshape(-1,1))))

# raw_data = yfinance.download(tickers = "^GSPC", start = "2025-10-1", 
#                               end = "2025-11-1", interval = "1d")
# print(raw_data)

raw_data2 = yfinance.download('MO GOOGL', start = "2016-1-1", 
                              end = "2026-1-1", interval = "1d").get('High')
bruh = (raw_data2.shift(-1).apply(np.log) - raw_data2.shift(1).apply(np.log))[1:-1]
print(bruh)

# print(raw_data2.axes)
# print(raw_data2.get('High').columns)

# raw_data2 = raw_data2.get('High')
# print(raw_data2.get([]))
# print(raw_data2.drop(["ZBRA", "MO"], axis = 1, inplace= False))
# print(raw_data2)
# print(np.array(raw_data))
# with open('snp500new.txt', 'r') as bruh:
#     stocks = [line.strip().upper() for line in bruh]
#     stocks.append('^GSPC')

# bruh = yfinance.download(tickers=stocks, start = "2016-1-1", end = "2026-1-1", interval = "1d")

# print(bruh.axes)

def snp500pred(stocks, number):
    raw_data = yfinance.download(stocks, start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
    daily_difference_of_logs = (raw_data.shift(-1).apply(np.log) - raw_data.shift(1).apply(np.log))[1:-1]
    
    snp = yfinance.download('^GSPC', start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
    daily_difference_of_logs_snp = (snp.shift(-1).apply(np.log) - snp.shift(1).apply(np.log))[1:-1]
    
    best_explanatory = []
    explanatory_betas = []

    for n in range(number):

        explanatory_stocks = daily_difference_of_logs.get(best_explanatory)
        remaining_stocks = daily_difference_of_logs.drop(best_explanatory, axis = 1, inplace = False)

        best_r2 = -1
        best_beta = 0
        best_stock = ""

        for stock in remaining_stocks.columns:
            
            reg = sklearn.linear_model.LinearRegression()
            reg.fit(remaining_stocks.get(explanatory_stocks.append(stock), daily_difference_of_logs_snp))
            
            r2 = sklearn.metrics.r2_score(daily_difference_of_logs_snp, reg.predict(explanatory_stocks.append(stock)))

            if r2 > best_r2:
                best_r2 = r2
                best_beta = reg.coef_
                best_stock = stock

        best_explanatory = best_explanatory.append(best_stock)
        explanatory_betas = explanatory_betas.append(best_beta)

    return [best_explanatory, explanatory_betas]
        






