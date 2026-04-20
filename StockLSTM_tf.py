import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime

start_date = datetime.datetime(2021, 4, 15)
end_date = datetime.datetime(2026, 4, 15)
stock = yf.Ticker('SNT.WA')
data = stock.history(start = start_date, end = end_date)

df = pd.DataFrame(data)
df = df[['Open', 'Close']]

def df_to_windowed_df(dataframe, n=3):
    df_windowed = dataframe[['Close']].copy()
    df_windowed.columns = ['Target']

    for i in range(1, n + 1):
        df_windowed[f'Target-{i}'] = df_windowed['Target'].shift(i)

    df_windowed = df_windowed.dropna()

    column_order = [f'Target-{i}' for i in range(n, 0, -1)] + ['Target']
    df_windowed = df_windowed[column_order]

    return df_windowed

windowed_df = df_to_windowed_df(df)

def windowed_df_to_X_y(windowed_df):
    dates = windowed_df.index.to_numpy()

    df_as_np = windowed_df.to_numpy()

    X = df_as_np[:, :-1]
    y = df_as_np[:, -1]

    X = X.reshape((len(dates), X.shape[1], 1))

    return dates, X.astype(np.float32), y.astype(np.float32)

dates, X, y = windowed_df_to_X_y(windowed_df)

q_80 = int(len(dates) * 0.8)
q_90 = int(len(dates) * 0.9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

model = Sequential([layers.Input((3,1)),
                layers.LSTM(64),
                layers.Dense(32, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)])

model.compile(loss = 'mse',
              optimizer=Adam(learning_rate = 0.001),
              metrics=['mae'])

model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 100)

train_predictions = model.predict(X_train).flatten()
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Train pred.', 'Train observations'])

val_predictions = model.predict(X_val).flatten()
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation pred.', 'Validation observations'])

test_predictions = model.predict(X_test).flatten()
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Test pred.', 'Test observations'])
plt.show()








