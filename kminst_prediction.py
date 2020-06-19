import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('kminst_dropout.h5')

layer5 = model.layers[7]
print(f"Layer name: {layer5.name}")

test_ds = tfds.load('kmnist', split='test')

test_images = []
test_labels = []

for i in test_ds:
    test_images.append(i['image'].numpy())
    test_labels.append(i['label'].numpy)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# random_indx = np.random.randint((len(test_labels)))
#
# print(f"Predicted label {(np.argmax(model(np.expand_dims(test_images[random_indx], axis = 0))))}")
# print(f"Actual label {test_labels[random_indx]}")

print(np.argmax(model.predict(test_images[:32]), axis = 1))

print(layer5.get_weights()[0].shape)
