import pandas as pd
import sklearn
import yfinance
import numpy as np
# import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import re
print(patsy.__version__)

complete_stocks = pd.read_csv("out.csv").set_index("Date")
complete_stocks.index = pd.to_datetime(complete_stocks.index)

class indexPredictor:

    # Struct which defines an index predicting model, defined by explanatory stocks and their betas.
    class indexPredictorModel:
        def __init__(self, index, explanatoryStocks, explanatoryBetas, interval):
            self.index = index
            self.explanatoryStocks = explanatoryStocks
            self.explanatoryBetas = explanatoryBetas
            self.interval = interval

        def mToString(self):
            return ("Index: " + self.index + "\nExplanatory stocks: " + " ".join(self.explanatoryStocks) +
                   "\nExplanatory betas: " + np.array2string(self.explanatoryBetas))

    #Note: Stocks will include indexes such as S&P. stockIndexData is entirely static, only ever read
    def __init__(self, stocks, indexes, start_ = "2016-1-1", end_ = "2026-1-1", interval_ = "1d", type = 'High', tester = False):
        if (not tester):
            self._stockData = yfinance.download(stocks, start = start_, end = end_, interval = interval_).get(type)
            self._indexData = yfinance.download(indexes, start = start_, end = end_, interval = interval_).get(type)
        else:
            self._stockData = complete_stocks
            self._indexData = yfinance.download("^GSPC", start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')


    #Given a set of stocks, a target index, and a number of explanatory stocks to return, returns an indexPredictorModel.
    def buildIndexPredictor(self, index, stocks, number, interval = None, allStocks = False):

        if (allStocks):
            stock_raw_data = self._stockData
        else:
            stock_raw_data = self._stockData.get(stocks)
        stock_diff_logs = (stock_raw_data.shift(-1).apply(np.log) - stock_raw_data.shift(1).apply(np.log))[1:-1]
        
        index_raw_data = self._indexData.get(index)
        index_diff_logs = (index_raw_data.shift(-1).apply(np.log) - index_raw_data.shift(1).apply(np.log))[1:-1]
        
        if (interval != None):
            stock_diff_logs = stock_diff_logs.loc[interval]
            index_diff_logs = index_diff_logs.loc[interval]

        best_explanatory = pd.DataFrame()
        explanatory_betas = []

        for n in range(number):

            print(n)

            best_r2 = -1
            best_beta = 0
            best_stock = ""

            for stock_ in stock_diff_logs:

                stock = stock_diff_logs.get(stock_)

                reg = sklearn.linear_model.LinearRegression(fit_intercept = False)
                if(n == 1):
                    print(best_explanatory)
                    print(stock)
                reg.fit(pd.concat([best_explanatory, stock], axis = 1), index_diff_logs)
                
                r2 = sklearn.metrics.r2_score(index_diff_logs, reg.predict(pd.concat([best_explanatory, stock], axis = 1)))

                if r2 > best_r2:
                    best_r2 = r2
                    best_beta = reg.coef_
                    best_stock = stock_

            best_explanatory = pd.concat([best_explanatory, stock_diff_logs.get(best_stock)], axis = 1)
            explanatory_betas = best_beta
            stock_diff_logs.drop(best_stock, axis = 1)

        return self.indexPredictorModel(index, best_explanatory.columns.to_list(), explanatory_betas, interval)

    def buildConstrainedPredictor(self, index, stocks, number, interval = None, allStocks = False):

        if (allStocks):
            stock_raw_data = self._stockData
        else:
            stock_raw_data = self._stockData.get(stocks)
        stock_diff_logs = (stock_raw_data.shift(-1).apply(np.log) - stock_raw_data.shift(1).apply(np.log))[1:-1]
        
        index_raw_data = self._indexData.get(index)
        index_diff_logs = (index_raw_data.shift(-1).apply(np.log) - index_raw_data.shift(1).apply(np.log))[1:-1]
        
        if (interval != None):
            stock_diff_logs = stock_diff_logs.loc[interval]
            index_diff_logs = index_diff_logs.loc[interval]

        best_explanatory = []

        # The constrained GLM thing doesn't like special characters
        cleaned_index = re.sub(r'[^a-zA-Z0-9]', '', index)
        index_diff_logs.name = cleaned_index

        temp_columns = stock_diff_logs.columns.to_list()

        for j in range(len(temp_columns)):
            temp_columns[j] = re.sub(r'[^a-zA-Z0-9]', '', temp_columns[j])
        
        stock_diff_logs.columns = temp_columns

        combined_data = pd.concat([stock_diff_logs, index_diff_logs], axis = 1)

        for n in range(number):

            print(n)

            best_r2 = -10000
            best_beta = 0
            best_stock = ""

            for stock_ in stock_diff_logs:
                
                if n == 0:
                    new_model = sklearn.linear_model.LinearRegression()
                    new_model.intercept_ = 0
                    new_model.coef_ = np.array([1])

                    r2 = sklearn.metrics.r2_score(index_diff_logs, new_model.predict(pd.DataFrame(stock_diff_logs.get(stock_))))
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_beta = 1
                        best_stock = stock_
                    
                else:

                    if stock_ not in best_explanatory:
                        formula_ = cleaned_index + " ~ " + stock_
                        constraint_formula = stock_

                        for explanatory in best_explanatory:
                            formula_ = formula_ + " + " + explanatory
                            constraint_formula = constraint_formula + " + " + explanatory

                        formula_ = formula_ + " - 1"
                        constraint_formula = constraint_formula + " = 1"

                        print(formula_)

                        reg_constrained = smf.glm(formula = formula_, data = combined_data, family = sm.families.Gaussian()).fit_constrained(constraint_formula)
                        print(reg_constrained.summary())

                        r2 = reg_constrained.pseudo_rsquared()

                        if r2 > best_r2:
                            best_r2 = r2
                            best_beta = np.array(reg_constrained.params)
                            best_stock = stock_

            best_explanatory.append(best_stock)
            explanatory_betas = best_beta
            print(best_r2)

        return self.indexPredictorModel(index, best_explanatory, explanatory_betas, interval)

    # def modelQualityOfFit(self, model, interval = None):

    #     new_model = sklearn.linear_model.LinearRegression()
    #     new_model.intercept_ = 0
    #     new_model.coef_ = model.explanatoryBetas
        
    #     index_raw_data = self._indexData.get(model.index)
    #     index_diff_logs = (index_raw_data.shift(-1).apply(np.log) - index_raw_data.shift(1).apply(np.log))[1:-1]

    #     # if (interval != None):
    #     #     index_interval = self._indexData.loc(interval)
    #     #     stock_interval = self._stockData.loc(interval)

    #     #     return sklearn.metrics.r2_score(index_diff_logs, new_model.predict(stock_interval.get(model.explanatoryStocks)))
    
    #     return sklearn.metrics.r2_score(index_diff_logs, new_model.predict(self._stockData.get(model.explanatoryStocks)))
    
    # def purchaseStocks(self, model, date, moneySpent):

    #     stock_raw_data = self._stockData.get(model.explanatoryStocks)

    #     stock_value = stock_raw_data.get(date)

    #     ratios = np.linalg.norm (model.explanatoryBetas) * moneySpent

    #     return stock_value * ratios

    def graphExample(self, model):

        stock_raw_data = self._stockData.get(model.explanatoryStocks)
        print(stock_raw_data)
        num_purchased = (1000*model.explanatoryBetas)/stock_raw_data.head(1)
        print(num_purchased)

        index_raw_data = self._indexData.get(model.index)
        print(index_raw_data)
        index_num_purchased = 1000/index_raw_data.head(1)
        print(index_num_purchased)

        print(stock_raw_data.iloc[-1])
        print(stock_raw_data.iloc[-1] * num_purchased)


        # print(index_raw_data)
        # print(index_num_purchased)

        # print(stock_raw_data)
        # print(num_purchased)
        # print(stock_raw_data*num_purchased.to_numpy())
        # print((stock_raw_data*num_purchased.to_numpy()).sum(axis=1))

        # print((index_raw_data*index_num_purchased.to_numpy()).rename("SnP"))
        # print(pd.merge(((stock_raw_data*num_purchased.to_numpy()).sum(axis=1)).rename("Stocks"), (index_raw_data*index_num_purchased.to_numpy()).rename("SnP"), on=["Date"]))
        # (stock_raw_data*num_purchased.to_numpy()).sum(axis=1).plot()

        # plt.show()


temp = indexPredictor(0,0, tester=True)
#bruh = temp.buildConstrainedPredictor("^GSPC",0,3,allStocks=True)
bruh = temp.indexPredictorModel("^GSPC", ["BRK-B", "MSFT", "APH"], np.array([0.24381594,  0.4359183,  0.32026576]), None)
print(bruh.mToString())
temp.graphExample(bruh)