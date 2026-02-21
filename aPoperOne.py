import pandas as pd
import sklearn
import yfinance
import numpy as np

def snp500pred(stocks, target, number):
    
    try:
        stocks.remove(target)
    except ValueError:
        pass

    raw_stock_data = yfinance.download(stocks, start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
    daily_difference_of_logs = (raw_stock_data.shift(-1).apply(np.log) - raw_stock_data.shift(1).apply(np.log))[1:-1]
    
    raw_target_data = yfinance.download(target, start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
    target_daily_difference_of_logs = (raw_target_data.shift(-1).apply(np.log) - raw_target_data.shift(1).apply(np.log))[1:-1]
    
    best_explanatory = pd.DataFrame()
    explanatory_betas = []

    for n in range(number):

        best_r2 = -1
        best_beta = 0
        best_stock = ""

        for stock_ in daily_difference_of_logs:
            stock = daily_difference_of_logs.get(stock_)

            reg = sklearn.linear_model.LinearRegression()
            reg.fit(pd.concat([best_explanatory, stock], axis = 1), target_daily_difference_of_logs)
            
            r2 = sklearn.metrics.r2_score(target_daily_difference_of_logs, reg.predict(pd.concat([best_explanatory, stock], axis = 1)))

            if r2 > best_r2:
                best_r2 = r2
                best_beta = reg.coef_
                best_stock = stock_

        best_explanatory = pd.concat([best_explanatory, daily_difference_of_logs.get(best_stock)], axis = 1)
        explanatory_betas.append(best_beta)
        daily_difference_of_logs.drop(best_stock, axis = 1)

    return [best_explanatory.columns.to_list(), explanatory_betas]

print(snp500pred(["GOOGL", "MO", "MMM"], "^GSPC", 2))