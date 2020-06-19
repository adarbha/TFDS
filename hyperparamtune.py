import tensorflow as tf
import kerastuner as kt
import numpy as np
import sys

from sklearn.preprocessing import StandardScaler

from kerastuner import HyperModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Dropout

from tensorflow.keras.datasets import boston_housing

from kerastuner.tuners.randomsearch import RandomSearch
from kerastuner.tuners.bayesian import  BayesianOptimization

print(f"TF version: {tf.__version__}")
print(f"Keras tuner version: {kt.__version__}")

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Scaling operations
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = tf.keras.models.Sequential()

# model.add(Lambda(lambda x: x / max_x_train, input_shape = (x_train.shape[1],)))
model.add(Dense(32, activation = 'relu', input_shape=(x_train.shape[1],)))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'sgd', metrics = ['mse'])

model.fit(x = x_train, y = y_train, batch_size = 32, epochs = 10, validation_data = (x_test, y_test))

model.evaluate(x_test, y_test)

class RegressionHypermodel(HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(
                  units=hp.Int('units', 8, 64, 4, default=8),
                  activation = hp.Choice('a_fn', ['relu', 'tanh']),
                  input_shape = self.input_shape)
        )

        model.add(
            Dense(units=hp.Int('units', 16, 64, 4, default=16),
                  activation=hp.Choice('a_fn', ['relu', 'tanh'])
        ))

        model.add(
            Dropout(rate = hp.Float('dropput_rate', min_value=0, max_value=0.1, default = 0.005, step=0.01))
        )

        model.add(Dense(1))

        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=['mse'])

        return model


hypermodel = RegressionHypermodel((x_train.shape[1],))

tuner_bo = BayesianOptimization(hypermodel,
                        objective='mse',
                        max_trials=10,
                        executions_per_trial=2,
                        overwrite=True)

tuner_bo.search(x_train, y_train, epochs = 10, validation_split = 0.1)

best_model = tuner_bo.get_best_models(1)[0]

print(tuner_bo.get_best_hyperparameters(1)[0].values)
print(best_model.evaluate(x_test, y_test))


