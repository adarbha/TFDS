import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ds, info = tfds.load('kmnist', split=['train', 'test'], with_info=True)

train_ds = ds[0]
test_ds = ds[1]

images_train = []
labels_train = []

for item in train_ds:
    images_train.append(item['image'].numpy())
    labels_train.append(item['label'].numpy())

images_train = np.array(images_train)
labels_train = np.array(labels_train)

images_test = []
labels_test = []

for item in test_ds:
    images_test.append(item['image'].numpy())
    labels_test.append(item['label'].numpy())

images_test = np.array(images_test)
labels_test = np.array(labels_test)

print(f"Number of train images {len(images_train)}")
print(f"Number of test images {len(images_test)}")

# Model creation
model = Sequential()
model.add(Lambda(lambda x: x / 255, input_shape=(28, 28, 1)))
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Prepare the data for the model
image_datagen = ImageDataGenerator()

train_generator = image_datagen.flow(x=images_train, y=labels_train)
validation_generator = image_datagen.flow(x=images_test, y=labels_test)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy']
              )

history = model.fit(train_generator, epochs = 30, validation_data=validation_generator, callbacks=[callback])

## Plot the results
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
epochs_ = len(history.history['loss'])
axes[0].plot(range(epochs_), history.history['loss'], label='training')
axes[0].plot(range(epochs_), history.history['val_loss'], label='validation')
axes[0].legend()

axes[1].plot(range(epochs_), history.history['accuracy'], label='training')
axes[1].plot(range(epochs_), history.history['val_accuracy'], label='validation')
axes[1].legend()

plt.savefig('kminst_lenet_dropout.png')

# Save the model
model.save('kminst_dropout.h5')
