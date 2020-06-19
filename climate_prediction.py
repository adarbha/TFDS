import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model


def prepare_data_for_pred(pred_array, window_size):
    # ds = tf.expand_dims(pred_array, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(pred_array)
    ds = ds.window(window_size, 1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(16).prefetch(1)
    return ds


df_test = pd.read_csv("climate_data/DailyDelhiClimateTest.csv", usecols=[1, 2, 3, 4])
test_data_arr = df_test.to_numpy(dtype=np.float)
window_size = 5

test_ds_batch = prepare_data_for_pred(test_data_arr, window_size)
model = load_model('climate_pred.h5')

test_batch = test_ds_batch.as_numpy_iterator()

preds = []

for item in test_batch:
    preds.extend(model(item).numpy()[:,-1])

print(len(preds))
print(len(test_data_arr[:,0]))


plt.figure(figsize = (10, 8))

actual_climate = test_data_arr[:, 0][window_size - 1:]
plt.plot(range(len(actual_climate)), actual_climate, label ='actual')
plt.plot(range(len(actual_climate)), preds, label ='predicted')
plt.legend()
plt.grid()
plt.savefig('climate_forecast.png')

print(f"MAE: {tf.keras.metrics.mae(actual_climate, preds)}")
