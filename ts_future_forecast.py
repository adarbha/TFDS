import tensorflow as tf
import csv
import matplotlib.pyplot as plt

import numpy as np

model = tf.keras.models.load_model('forecast_model.h5')


# Import the two csv files as a dataset
open_price = []
with open('stock_time_series/amd_stock.csv', 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)
    for row in csvreader:
        open_price.append(float(row[1]))

open_price = np.array(open_price, dtype=np.float)[6999:]
window_size = 30

dataset = tf.data.Dataset.from_tensor_slices(open_price)

ds = tf.expand_dims(open_price, axis = -1)
ds = tf.data.Dataset.from_tensor_slices(ds)
ds = ds.window(window_size, 1, drop_remainder = True)
ds = ds.flat_map(lambda w:w.batch(window_size))

ds = ds.batch(16).prefetch(1)

rnn_forecast = model.predict(ds)

print(rnn_forecast[:,-1,-1].shape)

plt.figure(figsize = (10, 8))

open_price_x = open_price[window_size - 1:]
plt.plot(range(len(open_price_x)), open_price_x, label = 'actual')
plt.plot(range(len(open_price_x)), rnn_forecast[:,-1,-1], label = 'predicted')
plt.legend()
plt.grid()
plt.savefig('forecast.png')

print(f"MAE: {tf.keras.metrics.mae(open_price_x, rnn_forecast[:,-1,-1])}")