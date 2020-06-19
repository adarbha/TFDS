import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Lambda, Conv1D
print(tf.__version__)

df_train = pd.read_csv("climate_data/DailyDelhiClimateTrain.csv", usecols=[1,2,3,4])
df_test = pd.read_csv("climate_data/DailyDelhiClimateTest.csv", usecols = [1,2,3,4])

# plt.figure()
# plt.plot(df_train.iloc[:,0])
# plt.grid()
# plt.savefig('temps.png')


train_data_arr = df_train.to_numpy(dtype= np.float)
test_data_arr = df_test.to_numpy(dtype = np.float)

print(f"Train data shape: {train_data_arr.shape}")
print(f"Test data shape: {test_data_arr.shape}")


train_ds = tf.data.Dataset.from_tensor_slices(train_data_arr)
test_ds = tf.data.Dataset.from_tensor_slices(test_data_arr)

window_size = 5
shuffle_buffer = 1024
batch_size = 128

def get_windowed_dataset(ds, window_size, shuffle_buffer, batch_size):
    ds = ds.window(window_size + 1, 1, drop_remainder=True) # This is for windowing
    ds = ds.flat_map(lambda w:w.batch(window_size + 1)) # This is to convert each window to a dataset and make the output flat
    ds = ds.shuffle(shuffle_buffer) # This just shuffles the data in the shuffle buffle
    ds = ds.map(lambda x:(x[:-1], x[-1:,0])) # This creates Xs and Ys
    ds = ds.batch(batch_size).prefetch(1) # This creates an iter for batch loading
    return ds

train_ds_batched = get_windowed_dataset(train_ds, window_size, shuffle_buffer, batch_size)
test_ds_batched = get_windowed_dataset(test_ds, window_size, shuffle_buffer, batch_size)


model = Sequential([Conv1D(filters=64, kernel_size = 5, padding = 'causal', activation = 'relu', input_shape = [None, 4]),
    LSTM(128, return_sequences = True),
    LSTM(64, return_sequences = False),
    Dense(40, activation = 'relu'),
    Dense(30, activation = 'relu'),
    Dense(1),
    Lambda(lambda x: x * 40)])

model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4), metrics = ['mae'])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))

history = model.fit(train_ds_batched, epochs = 300)

model.save('climate_pred.h5')

# Plot learning rate

plt.figure(figsize=(10, 8))

# plt.semilogx(history.history['lr'], history.history['loss'])
# plt.grid()
# plt.savefig('learning_rate.png')
plt.plot(history.history['mae'])
plt.grid()
plt.savefig('mae.png')
