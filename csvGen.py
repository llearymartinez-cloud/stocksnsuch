import pandas as pd
import sklearn
import yfinance
import numpy as np

# raw_stock_data = yfinance.download(["GOOGL", "MO", "MMM"], start = "2016-1-1", end = "2026-1-1", interval = "1d").get('High')
# raw_stock_data.to_csv("out.csv")
print(pd.read_csv("out.csv").set_index("Date"))