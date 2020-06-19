import matplotlib.pyplot as plt

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Make a stream out of data downloaded from tensorflow
train_ds, val_ds = tfds.load('imdb_reviews', split=['train','test'])

# for i in train_ds.take(3).as_numpy_iterator():
#     print(i)

# Create sentences list and labels list used for sending it to
# Remove stop words from sentences when adding to lists
def create_sent_labels(ds):
    stop_words =  [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
    sentences = []
    labels = []
    for i in ds.as_numpy_iterator():
        temp_sent = str(i['text'])
        for word in stop_words:
            if word in (temp_sent):
                token = " " + word + " "
                temp_sent = temp_sent.replace(token, " ")
        sentences.append(temp_sent)
        labels.append(i['label'])
    return sentences, labels


train_sentences, train_labels = create_sent_labels(train_ds)
val_sentences, val_labels = create_sent_labels(val_ds)

print(f"Length of train sequences {len(train_sentences)}")
print((f"Length of val sequences {len(val_sentences)}"))


# Create a tokenizer
VOCAB_SIZE = 10000
MAX_LEN = 1600
EMBED_DIM = 16

tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token="<oov>")

# Fit sentences to tokenizer and create sequences
tokenizer.fit_on_texts(train_sentences)

train_seq = tokenizer.texts_to_sequences(train_sentences)
test_seq = tokenizer.texts_to_sequences(val_sentences)

padded_train_seq = pad_sequences(train_seq, padding="pre", maxlen = MAX_LEN)
padded_test_seq = pad_sequences(train_seq, padding="pre", maxlen = MAX_LEN)

# Build a model with embedding layer
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length = MAX_LEN))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(10, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

# Compile
model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics = ['accuracy'])

# train
epochs_ = 5
history = model.fit(padded_train_seq, np.array(train_labels), epochs = epochs_, validation_data=(padded_test_seq, np.array(val_labels)))

# plot
fig, axes = plt.subplots(2,1)
fig.figsize = (10,8)
axes[0].plot(range(epochs_), history.history['loss'], label = 'training')
axes[0].plot(range(epochs_), history.history['val_loss'], label = 'validation')
axes[0].grid()
axes[0].legend()

axes[1].plot(range(epochs_), history.history['accuracy'], label = 'training')
axes[1].plot(range(epochs_), history.history['val_accuracy'], label = 'validation')
axes[1].grid()
axes[1].legend()

plt.savefig('NLP_binary.png')