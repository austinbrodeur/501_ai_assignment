from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd

train_file_path = "heart_train.csv"
test_file_path = "heart_test.csv"

LABEL_COLUMN = "chd"
LABELS = [0, 1]
NUMERIC_FEATURES = ['row.names', 'sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
CATEGORIES = {
    'famhist': ['Present', 'Absent']
}

np.set_printoptions(precision=3, suppress=True)


class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels



def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True
    )
    return dataset


def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


def make_model(preprocessing_layer):

    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model

def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


def main():
    raw_train_data = get_dataset(train_file_path)
    raw_test_data = get_dataset(test_file_path)

    packed_train_data = raw_train_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

    packed_test_data = raw_test_data.map(
        PackNumericFeatures(NUMERIC_FEATURES))

    desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

    MEAN = np.array(desc.T['mean'])
    STD = np.array(desc.T['std'])

    normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer,
                                                      shape=[len(NUMERIC_FEATURES)])
    numeric_columns = [numeric_column]

    numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)

    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        categorical_columns.append(tf.feature_column.indicator_column(cat_col))

    categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

    model = make_model(preprocessing_layer)

    train_data = packed_train_data.shuffle(500)
    test_data = packed_test_data

    model.fit(train_data, epochs=20, verbose=2)
    test_loss, test_accuracy = model.evaluate(test_data)
    print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

main()