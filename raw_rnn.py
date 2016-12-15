from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import functools
from data_processing import load_csvdata
import matplotlib.pyplot as plt


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

def linear(self, x, name, size):
    W = tf.get_variable(name+"/W", [x.get_shape()[1], size])
    b = tf.get_variable(name+"/b", [size], initializer=tf.zeros_initializer)

    return tf.matmul(x, W) + b

TIMESTEPS = 10

class RNN_LR(object):

    def __init__(self, data, target, dropout, num_hidden=10, num_layers=5):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error = None
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cell = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=self.dropout)
        cells = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers)
        data_ = tf.unpack(self.data, axis=1)
        output, _ = tf.nn.rnn(cells, data_, dtype=tf.float32)
        # Softmax layer.
        #max_length = int(self.target.get_shape()[1])
        #num_classes = int(self.target.get_shape()[2])
        h1 = tf.contrib.layers.relu(output[-1], 10)
        h2 = tf.contrib.layers.relu(h1, 10)
        # Flatten to apply same weights to all time steps.
        #output = tf.reshape(output, [-1, self._num_hidden])
        #output = output[-1]
        #weight, bias = self._weight_and_bias(self._num_hidden, 1)
        weight, bias = self._weight_and_bias(10, 1)
        prediction = tf.matmul(h2, weight) + bias
        #prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        mse = tf.losses.mean_squared_error(self.prediction, self.target)
        return mse

    @lazy_property
    def optimize(self):
        learning_rate = 5e-4
        optimizer = "Adam"
        train_op = tf.contrib.layers.optimize_loss(
            loss=self.cost,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer=optimizer)
        return train_op

    #@lazy_property
    #def error(self):
    #    mistakes = tf.not_equal(
    #        tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
    #    return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def process_data(data, timesteps, targets=False):
    output = []
    for i in range(len(data) - timesteps):
        if targets:
            try:
                output.append(data.iloc[i + timesteps].as_matrix())
            except AttributeError:
                output.append(data.iloc[i + timesteps])
        else:
            data_ = data.iloc[i: i+timesteps].as_matrix()
            output.append(data_ if len(data_.shape)>1 else [[i] for i in data_])

    return np.array(output, dtype=np.float32)

def load_data(filename, target_name, timesteps, split=None):
    df = pd.read_csv(filename, names=["date", "in_id", "out_id", "calls", "duration"], parse_dates=['date'])
    df = df['calls'].groupby(df.date).sum()
    df = df.reset_index()
    df = df.set_index("date")

    if split:
        cutoff = int(len(df) * split)
        df_test = df.iloc[:cutoff]
        df_val = df.iloc[cutoff:]
        features_test = process_data(df_test, timesteps)
        targets_test = process_data(df_test, timesteps, targets=True)
        features_val = process_data(df_val, timesteps)
        targets_val = process_data(df_val, timesteps, targets=True)

        return features_test, features_val, targets_test, targets_val

    else:
        features = process_data(df, timesteps)
        targets = process_data(df, timesteps, targets=True)

        return features, targets


class Set(object):

    def __init__(self,filename, split=None):

        X, Y = load_data(filename, "calls", TIMESTEPS, split=split)

        self._features = X
        self._targets = Y
        self._num_examples = len(X)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
      return self._features

    @property
    def targets(self):
      return self._targets

    @property
    def num_examples(self):
      return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
            end = self._index_in_epoch
        end = self._index_in_epoch
        return self._features[start:end], self._targets[start:end]


class Dataset(object):

    def __init__(self, train_file, test_file):
        self._train = Set(train_file)
        self._test = Set(test_file)

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test


if __name__ == '__main__':

    dataset = Dataset("data/SET1V_01_bs2.csv", "data/SET1V_02_bs2.csv")

    data = tf.placeholder(tf.float32, [None, TIMESTEPS, 1])
    target = tf.placeholder(tf.float32, [None, 1])
    dropout = tf.placeholder(tf.float32)
    model = RNN_LR(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        for _ in range(7):
            features, targets = dataset.train.next_batch(100)
            _, error = sess.run([model.optimize, model.cost], {
                data: features, target: targets, dropout: 1})
        features, targets = dataset.test.next_batch(100)
        error = sess.run(model.cost, {
            data: features, target: targets, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch, error))

    predictions = []
    for x in dataset.test.features:
        buffer_target = [[0]]
        pred = sess.run(model.prediction, {data: [x], target: buffer_target, dropout: 1})
        predictions.append(pred.flatten()[0])
    predictions = np.array(predictions)
    print(predictions)
    rmse = np.sqrt(((predictions - dataset.test.targets.flatten()) ** 2).mean(axis=0))
    print("RMSE : %f" % rmse)

    from pymongo import MongoClient
    from bson.objectid import ObjectId
    import dateutil.parser
    import datetime
    import matplotlib.dates as mdates

    df = pd.read_csv("data/SET1V_02_bs2.csv", names=["date", "in_id", "out_id", "calls", "duration"], parse_dates=['date'])
    df = df['calls'].groupby(df.date).sum()
    df = df.reset_index()
    df = df.set_index("date")
    all_dates = df.index.get_values()
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()

    #predicted_values = test_predictions.flatten() #already subset
    predicted_values = predictions
    predicted_dates = all_dates[len(all_dates)-len(predicted_values):len(all_dates)]
    predicted_series = pd.Series(predicted_values, index=predicted_dates)
    plot_predicted, = ax.plot(predicted_series, label='predicted (c)')

    test_values = dataset.test.targets.flatten()
    test_dates = all_dates[len(all_dates)-len(test_values):len(all_dates)]
    test_series = pd.Series(test_values, index=test_dates)
    plot_test, = ax.plot(test_series, label=' (c)')

    xfmt = mdates.DateFormatter('%b %d %H')
    ax.xaxis.set_major_formatter(xfmt)

    # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
    plt.title('PDX Weather Predictions for 2016 vs 2015')
    plt.legend(handles=[plot_predicted, plot_test])
    plt.show()
