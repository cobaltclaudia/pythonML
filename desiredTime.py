import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ffn
from pandas_datareader import data as pdr
import yfinance as yf
import datetime

yf.pdr_override()


# determine if positive or negative
def computeClassification(actual):
    if actual > 0:
        return 1
    else:
        return -1


# desired period of time
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2018, 10, 29)
stonks = 'AMZN'

# pull data from yahoo finance
df = pdr.get_data_yahoo(stonks, start, end)
print("Current df ")
print(df)

# calculate daily returns
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['returns'].fillna(0)
df['returns_1'] = df['returns'].fillna(0)
df['returns_2'] = df['returns_1'].replace([np.inf, -np.inf], np.nan)
df['returns_final'] = df['returns_2'].fillna(0)
print(df['returns_final'].size)
print(df['returns_final'])

# apply computeClassification and show 1/-1 in adjacent column
df.iloc[:, len(df.columns) - 1] = df.iloc[:, len(df.columns) - 1].apply(computeClassification)
print(df.iloc[:, len(df.columns) - 1])
