import pandas as pd
import sqlalchemy
import csv
import json
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

# Read messages csv convert into numpy array - remove stop words while reading in sentences. Also read labels here
stop_words = ("a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
              "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
              "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
              "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves")

def remove_stop_words(sentence, stopwords = stop_words):
    url_regex = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    sentence = re.sub(url_regex, "url_placeholer", sentence)
    for word in stopwords:
        sentence = sentence.replace("".join([" ",word, " "]), " ")
    return sentence


# Read DisasterResponse.db as it contains cleaned data
engine = sqlalchemy.engine.create_engine("sqlite:///disaster_data/DisasterResponse.db")
df_chunks = pd.read_sql_query("SELECT * FROM msg_cat", engine, chunksize=1024)

# Create a corpus of sentences and convert them to sequences - Do this while chunks of dataframe are being read in Make sure even the labels are handled
sentences = []
labels = []

for chunk in df_chunks:
    for row in chunk.to_numpy():
        preprocessed_text = remove_stop_words(str(row[1]))
        sentences.append(remove_stop_words(preprocessed_text))
        labels.append(np.array(row[4:], dtype=np.float))

tokenizer = Tokenizer(num_words=10000, oov_token="<oov>")
tokenizer.fit_on_texts(sentences)
token_json = tokenizer.to_json()
with open('token.json', 'w') as f:
    json.dump(token_json, f)

sequences = tokenizer.texts_to_sequences(sentences)
max_len = max([len(i) for i in sequences])
padded = pad_sequences(sequences, maxlen=max_len, padding="pre", truncating="pre")
padded = np.nan_to_num(padded)


#  Convert labels
labels = np.nan_to_num(np.array(labels))


# Model building
VOCAB_SIZE = 10000
EMBED_SIZE = 16
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=max_len))
model.add(GRU(16))
model.add(Dense(32, activation='relu'))
model.add(Dense(36, activation='sigmoid'))


#  Model compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

#  Model fit
early_stopping = EarlyStopping('val_accuracy', patience = 4)

history = model.fit(padded, labels, epochs = 5, callbacks=[early_stopping], validation_split=0.1)


#  Model save
model.save('disaster.h5')

# evaluation plots
epochs_ = len(history.history['loss'])

fig, axes = plt.subplots(2,1)
axes[0].plot(range(epochs_), history.history['loss'], label='training')
axes[0].plot(range(epochs_), history.history['val_loss'], label='validation')
axes[0].legend()

axes[1].plot(range(epochs_), history.history['accuracy'], label='training')
axes[1].plot(range(epochs_), history.history['val_accuracy'], label='validation')
axes[1].legend()


plt.savefig('disaster.png')

