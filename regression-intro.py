import pandas as pd
import quandl, math, os
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

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
print(forecast_out)

# We are shifting the dataset by the forecase_out 'up' so we can determine
# the data for the future days
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# We are removing our 'label' column and setting it as a numpy array
# Note: drop only removes it in the return value, no in the actual object
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

# Skip in high frequency trading
X = preprocessing.scale(X)

y = np.array(df['label'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
# clf = svm.SVR(kernel='poly')
# Fit = train
clf.fit(X_train, y_train)

# Score = test
accuracy = clf.score(X_test, y_test)

print(accuracy)



