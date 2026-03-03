import pandas as pd
import sklearn
import yfinance
import numpy as np

class indexPredictor:

    # Struct which defines an index predicting model, defined by explanatory stocks and their betas.
    class indexPredictorModel:
        def __init__(self, index, explanatoryStocks, explanatoryBetas, interval):
            self.index = index
            self.explanatoryStocks = explanatoryStocks
            self.explanatoryBetas = explanatoryBetas
            self.interval = interval

    #Note: Stocks will include indexes such as S&P. stockIndexData is entirely static, only ever read
    def __init__(self, stocks, indexes, start_ = "2016-1-1", end_ = "2026-1-1", interval_ = "1d", type = 'High'):
        self._stockData = yfinance.download(stocks, start = start_, end = end_, interval = interval_).get(type)
        self._indexData = yfinance.download(indexes, start = start_, end = end_, interval = interval_).get(type)

    #Given a set of stocks, a target index, and a number of explanatory stocks to return, returns an indexPredictorModel.
    def buildIndexPredictorM(self, index, stocks, number, interval = None):

        stock_raw_data = self._stockData.get(stocks)
        stock_diff_logs = (stock_raw_data.shift(-1).apply(np.log) - stock_raw_data.shift(1).apply(np.log))[1:-1]
        
        index_raw_data = self._indexData.get(index)
        index_diff_logs = (index_raw_data.shift(-1).apply(np.log) - index_raw_data.shift(1).apply(np.log))[1:-1]
        
        if (interval != None):
            stock_diff_logs = stock_diff_logs.loc[interval]
            index_diff_logs = index_diff_logs.loc[interval]

        best_explanatory = pd.DataFrame()
        explanatory_betas = []

        final_r2 = 0

        for n in range(number):

            print(n)

            best_r2 = -1
            best_beta = 0
            best_stock = ""

            for stock_ in stock_diff_logs:

                stock = stock_diff_logs.get(stock_)

                reg = sklearn.linear_model.LinearRegression()
                reg.fit(pd.concat([best_explanatory, stock], axis = 1), index_diff_logs)
                
                r2 = sklearn.metrics.r2_score(index_diff_logs, reg.predict(pd.concat([best_explanatory, stock], axis = 1)))

                if r2 > best_r2:
                    best_r2 = r2
                    best_beta = reg.coef_
                    best_stock = stock_
                    final_r2 = r2

            best_explanatory = pd.concat([best_explanatory, stock_diff_logs.get(best_stock)], axis = 1)
            explanatory_betas = best_beta
            stock_diff_logs.drop(best_stock, axis = 1)

        return self.indexPredictorModel(index, best_explanatory, explanatory_betas, interval)
    
    def modelQualityOfFit(self, model, interval = None):\
    #need to account for the model interval
        stock_raw_data = self._stockData.get(model.stocks)
        stock_diff_logs = (stock_raw_data.shift(-1).apply(np.log) - stock_raw_data.shift(1).apply(np.log))[1:-1]
        
        index_raw_data = self._indexData.get(model.index)
        index_diff_logs = (index_raw_data.shift(-1).apply(np.log) - index_raw_data.shift(1).apply(np.log))[1:-1]
        
        if (interval != None):
            stock_diff_logs = stock_diff_logs.loc[interval]
            index_diff_logs = index_diff_logs.loc[interval]

        reg = sklearn.linear_model.LinearRegression()
        reg.fit(stock_diff_logs, index_diff_logs)


