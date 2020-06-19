import csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Lambda

# Import the two csv files as a dataset
open_price = []
with open('stock_time_series/amd_stock.csv', 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)
    for row in csvreader:
        open_price.append(float(row[1]))

open_price = np.array(open_price, dtype=np.float)[:6999]



# Plot the time series for a visual
# fig, axes = plt.subplots()
# fig.figsize = (10,8)
#
# axes.plot(range(len(open_price)), open_price)
# axes.grid()
# axes.set_title("AMD_stock_time_series")
# plt.savefig('time_series_amd.png')

# Create a function that creates a window of data and batches them for the model to predict
window_size = 30
shuffle_buffer = 10000
batch_size = 32

def create_window_dataset(series, window_size, shuffle_buffer, batch_size):
    series = tf.expand_dims(series, axis = -1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    ds = dataset.window(window_size  + 1, 1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda x: (x[:-1], x[1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


model = Sequential([Conv1D(filters=32, kernel_size = 5, padding = 'causal', activation = 'relu', input_shape = [None, 1]),
    LSTM(64, return_sequences = True),
    LSTM(64, return_sequences = True),
    Dense(30, activation = 'relu'),
    Dense(10, activation = 'relu'),
    Dense(1),
    Lambda(lambda x: x * 50)])

model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.SGD(learning_rate=4*1e-6), metrics = ['mae'])

train_dataset = create_window_dataset(open_price, window_size, shuffle_buffer, batch_size)

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))

history = model.fit(train_dataset, epochs = 100)

# Plot learning rate

plt.figure(figsize=(10, 8))

plt.semilogx(history.history['lr'], history.history['loss'])
plt.grid()
plt.savefig('learning_rate.png')



model.save('forecast_model_climate.h5')

