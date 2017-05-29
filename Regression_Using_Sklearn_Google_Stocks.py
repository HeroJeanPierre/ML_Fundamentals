import pandas as pd
import quandl, math, datetime
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
# This simple program will be used to predict 10% out what Adj. Close will be
# For GOOGL stock. The method that I will be usin is LinearRegression through
# Sklearns linear model library.

# Use quandl's database to import the csv for GOOGL stock
df = quandl.get('WIKI/GOOGL')
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Revalue the dataframe
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

# This is the column that will be predicted 01% out
forecast_col = 'Adj. Close'

# Fill rows that don't contain data with outliers.
df.fillna(-99999, inplace=True)

# How many data points that will be forecasted out
forecast_out = int(math.ceil(0.01*len(df)))

# Create the label (y) column
df['label'] = df[forecast_col].shift(-forecast_out)

# Create features and labels
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

# This is used to set the values for our training data
# and to randomize them.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# Jobs at -1 uses all processsors
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Running Linear Regression classifier...')
print(colored('Accuracy is: {}'.format(accuracy), 'green'))

forecast_set = clf.predict(X_lately)
print(forecast_set, forecast_out)

# Now we will print the plot
style.use('ggplot')
df['Forecast'] = np.nan

last_data = df.iloc[-1].name
last_unix = last_data.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





                         
