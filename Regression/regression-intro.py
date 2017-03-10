import pandas as pd
import quandl, math, os, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
# allows you to save the classifier
import pickle

style.use('ggplot')

# df = data frame
df = quandl.get("WIKI/GOOGL", authtoken=os.environ.get('QUANDL_TOKEN'))
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# High low percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# Percent change from open to close
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Getting the columns we are interested in.
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# What are we trying to find out.
forecast_col = 'Adj. Close'

# Making missing data more of an outlier
df.fillna(-99999, inplace=True)

# We are trying to predict out 10% out of the data
# So we would use the previous data to try to predict tomorrow data
forecast_out = int(math.ceil(0.01 * len(df)))

# We are shifting the dataset by the forecase_out 'up' so we can determine
# the data for the future days
df['label'] = df[forecast_col].shift(-forecast_out)

# We are removing our 'label' column and setting it as a numpy array
# Note: drop only removes it in the return value, no in the actual object
X = np.array(df.drop(['label'],1))
# Skip in high frequency trading
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
# clf = svm.SVR(kernel='poly')
# Fit = train
clf.fit(X_train, y_train)
with open('linearregression.pickle', 'wb') as f:
  pickle.dump(clf, f)


# Score = test
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
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

