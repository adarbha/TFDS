import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

ds, info = tfds.load('yelp_polarity_reviews/subwords8k', split='train', with_info=True)

train_size = 100000
VOCAB_SIZE = 8176
EMBED_SIZE = 16
padding = 'pre'
truncating = 'pre'

text = []
label = []

for i in ds.take(train_size):
    text.append(i['text'].numpy())
    label.append(i['label'].numpy())

max_len = max([len(i) for i in text])

print(f"Max length of a review: {max_len}")

#Build pad sequence using max length
padded_text = pad_sequences(text, maxlen=max_len, padding=padding, truncating=truncating)

#Preprocess the labels so that they can be processed by the model
label = pad_sequences(np.expand_dims(np.array(label), axis=-1), maxlen = 1)

#Build a model - Embedding layer should the first layer, followed by RNNs
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=max_len))
model.add(GRU(16))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#Model compile and print summary
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

# Model train implement earlystopping callback
early_stopping = EarlyStopping('val_accuracy', patience = 4)

history = model.fit(padded_text, label, epochs = 5, callbacks=[early_stopping], validation_split=0.1)

#Model save
model.save('yelp_8k.h5')

#Model plots
epochs_ = len(history.history['loss'])

fig, axes = plt.subplots(2,1)
axes[0].plot(range(epochs_), history.history['loss'], label='training')
axes[0].plot(range(epochs_), history.history['val_loss'], label='validation')
axes[0].legend()

axes[1].plot(range(epochs_), history.history['accuracy'], label='training')
axes[1].plot(range(epochs_), history.history['val_accuracy'], label='validation')
axes[1].legend()


plt.savefig('yelp_8k_eval_plots.png')





