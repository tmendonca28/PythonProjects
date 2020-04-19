from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print(f"Training entries {len(train_data)}. Labels {len(train_labels)}")
# print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()

word_index = {k: (v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, "?") for i in text])

# Pad each of the reviews
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["PAD"],
                                                        padding="post",
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["PAD"],
                                                        padding="post",
                                                        maxlen=256)

