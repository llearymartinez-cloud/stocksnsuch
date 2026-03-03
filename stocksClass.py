import pandas as pd
import sklearn
import yfinance
import numpy as np

class indexPredictor:

    # Struct which defines an index predicting model, defined by explanatory stocks and their betas.
    class indexPredictorModel:
        def __init__(self, index, explanatoryStocks, explanatoryBetas):
            self.index = index
            self.explanatoryStocks = explanatoryStocks
            self.explanatoryBetas = explanatoryBetas

    #Note: Stocks will include indexes such as S&P. stockIndexData is entirely static, only ever read
    def __init__(self, stocksAndIndex, start_ = "2016-1-1", end_ = "2026-1-1", interval_ = "1d", type = 'High'):
        self._stockIndexData = yfinance.download(stocksAndIndex, start = start_, end = end_, interval = interval_).get(type)

    #Given a set of stocks, a target index, and a number of explanatory stocks to return, returns an indexPredictorModel.
    def indexPredictor__(self, stocks, index, number, inplace = False):

        try:
            stocks.remove(index)
        except ValueError:
            pass

        stock_raw_data = np.transpose(self.stockData.get(stocks).to_numpy())
        stock_diff_logs = np.log(stock_raw_data[1:]) - np.log(stock_raw_data[:-1])
        
        index_raw_data = np.transpose(self.stockData.get(index).to_numpy())
        index_diff_logs = np.log(index_raw_data[1:]) - np.log(index_raw_data[:-1])
        
        best_explanatory = np.array([])
        explanatory_betas = np.array([])

        final_r2 = 0

        for n in range(number):

            print(n)

            best_r2 = -1
            best_beta = 0
            best_stock = 0

            for i in range()

                stock = daily_difference_of_logs.get(stock_)

                reg = sklearn.linear_model.LinearRegression()
                reg.fit(pd.concat([best_explanatory, stock], axis = 1), target_daily_difference_of_logs)
                
                r2 = sklearn.metrics.r2_score(target_daily_difference_of_logs, reg.predict(pd.concat([best_explanatory, stock], axis = 1)))

                if r2 > best_r2:
                    best_r2 = r2
                    best_beta = reg.coef_
                    best_stock = stock_
                    final_r2 = r2

            best_explanatory = pd.concat([best_explanatory, daily_difference_of_logs.get(best_stock)], axis = 1)
            explanatory_betas.append(best_beta)
            daily_difference_of_logs.drop(best_stock, axis = 1)

        return [best_explanatory.columns.to_list(), explanatory_betas[-1], final_r2]