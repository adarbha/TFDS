import numpy as np
from skimage.transform import resize

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image, label = tfds.as_numpy(tfds.load(
    'beans',
    split='train',
    batch_size=-1,
    as_supervised=True,
))

test_image, test_label = tfds.as_numpy(tfds.load(
    'beans',
    split='test',
    batch_size=-1,
    as_supervised=True
))

# Create a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Lambda(lambda x:tf.image.resize(x, (150,150))))
model.add(tf.keras.layers.Conv2D(32, (3,3), input_shape=(None, 150,150)))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

# Create a training Image generator - no augmentatio
image_datagen = ImageDataGenerator(rescale = 1/255)
image_generator = image_datagen.flow(x=image, y=label,
                                     batch_size = 16)

# Create a validation Image generator
validation_datagen = ImageDataGenerator(rescale = 1/255)
validation_generator = validation_datagen.flow(x=test_image, y = test_label,
                                               batch_size=8)

# Model compile
model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

# Model training
epochs_ = 30
history = model.fit(image_generator, epochs = epochs_, validation_data = validation_generator)

fig, axes = plt.subplots(2,1)
fig.figsize = (10,8)
axes[0].plot(range(epochs_), history.history['loss'], label = 'training')
axes[0].plot(range(epochs_), history.history['val_loss'], label = 'validation')
axes[0].grid()
axes[0].legend()

axes[1].plot(range(epochs_), history.history['accuracy'], label = 'training')
axes[1].plot(range(epochs_), history.history['val_accuracy'], label = 'validation')
axes[1].grid()
axes[0].grid()

plt.savefig('validation_beans.png')