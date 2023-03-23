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
df[df.columns[len(df.columns) - 1]] = df[df.columns[len(df.columns) - 1]].apply(computeClassification)
print(df.iloc[:, len(df.columns) - 1])

# convert float to int
# 2nd half is forward tested on
testData = df[-int((len(df) * 0.10)):]

# 1st half is trained on
trainData = df[-int((len(df) * 0.90)):]

# replace all inf with nan
testData_1 = testData.replace([np.inf, -np.inf], np.nan)
trainData_1 = trainData.replace([np.inf, -np.inf], np.nan)

# replace all nans with 0
testData_2 = testData_1.fillna(0)
trainData_2 = trainData_1.fillna(0)

# x is the list of features
data_x_train = trainData_2.iloc[:, 0:len(trainData_2.columns) - 1]

# y is the 1 or -1 value to be predicted
data_y_train = trainData_2.iloc[:, 0:len(trainData_2.columns) - 1]

# same thing for the test dataset
data_x_test = testData_2.iloc[:, 0:len(testData_2.columns) - 1]
data_y_test = testData_2.iloc[:, 0:len(testData_2.columns) - 1]


