import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from tensorflow.keras.applications import VGG16

from tensorflow.keras.callbacks import Callback

# Create a call back function
class accCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        '''Call this function when accuracy is more than 0.98'''
        if logs.get('acc') > 0.98:
            print(f"\nAccuracy is a above 0.98 - STOP EPOCHS IMPLEMENTED")
            self.model.stop_training = False

# Read files and display a random image with its image size
TRAIN_DIR = 'hand_signs_dataset/Train/'
TEST_DIR = 'hand_signs_dataset/Test/'
VGG_IMG_SIZE = (224,224)

# load a random image from A
image = load_img("".join([TRAIN_DIR, "A/", np.random.choice(os.listdir(TRAIN_DIR + "A/"))]), VGG_IMG_SIZE)
plt.figure()
plt.imshow(image)
plt.savefig('test.png')

# Create ImageData generator - no augmentation
image_datagen = ImageDataGenerator(rescale = 1/255)
train_image_generator = image_datagen.flow_from_directory(directory=TRAIN_DIR,
                                                          target_size=VGG_IMG_SIZE,
                                                          class_mode='categorical',
                                                          batch_size=32)

val_datagen = ImageDataGenerator(rescale=1/255)
val_image_generator = val_datagen.flow_from_directory(directory=TEST_DIR,
                                                        target_size=VGG_IMG_SIZE,
                                                        batch_size=32)


# Download VGG16 neuralnet for initial layers of training
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

print(f"Number of layers in the model {len(vgg16.layers)}")

# Freeze all the layers
for layer in vgg16.layers:
    layer.trainable = False

# Subset layers to include top N layers
TOP_N = 11
vgg_16_sub = Sequential()
layer_count = 0
for layer in vgg16.layers:
    vgg_16_sub.add(layer)
    layer_count+=1
    if layer_count > TOP_N:
        break


# Add new layers on top of layer subset from above
flatten = Flatten()(vgg_16_sub.output)
dense_1 = Dense(8, activation = 'relu')(flatten)
classifier = Dense(24, activation = 'softmax')(dense_1)

model = Model(inputs = vgg_16_sub.inputs, outputs = classifier)

print(model.summary())

# Model compile
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

# Model fit
epochs_ = 3
history = model.fit(train_image_generator, epochs = epochs_, validation_data = val_image_generator)

# Plots
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

plt.savefig('validation_transfer.png')