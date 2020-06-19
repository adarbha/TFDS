import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_datasets as tfds

ds = tfds.load('tiny_shakespeare', split = 'train')

# Get sentences
corpus = []
for i in ds.as_numpy_iterator():
    corpus.append(str(i['text']))

sentences = corpus[0].lower().split("\\n")[:10000]

# Create a tokenizer and get word_index
tokenizer = Tokenizer(oov_token="<oov>")
tokenizer.fit_on_texts(sentences)

number_of_words = len(tokenizer.word_index) + 1

# Convert sentences to sequences
input_sequences = []
for sent in sentences:
    sequence = (tokenizer.texts_to_sequences([sent])[0])
    for i in range(1, len(sequence) + 1):
        n_gram_seq = sequence[:i+1]
        input_sequences.append(n_gram_seq)

# Pad the sequences
max_length_ = max([len(i) for i in input_sequences])
padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_length_, padding="pre"))

# Create training and labels
training, labels = padded_sequences[:,:-1], padded_sequences[:,-1]

labels = tf.keras.utils.to_categorical(labels, num_classes=number_of_words)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(number_of_words, 100, input_length=max_length_ - 1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(number_of_words / 2, activation='relu'))
model.add(tf.keras.layers.Dense(number_of_words, activation = 'softmax'))

# Compile
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit
epochs_ = 20
history = model.fit(training, labels, epochs = epochs_, verbose = 1)

# plot
# fig, axes = plt.subplots(2,1)
# fig.figsize = (10,8)
# axes[0].plot(range(epochs_), history.history['loss'], label = 'training')
# axes[0].plot(range(epochs_), history.history['val_loss'], label = 'validation')
# axes[0].grid()
# axes[0].legend()
#
# axes[1].plot(range(epochs_), history.history['accuracy'], label = 'training')
# axes[1].plot(range(epochs_), history.history['val_accuracy'], label = 'validation')
# axes[1].grid()
# axes[1].legend()

# Save
model.save('text_generation_model.h5')