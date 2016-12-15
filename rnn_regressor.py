from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from data_processing import generate_data, lstm_model, load_csvdata
from pymongo import MongoClient
from bson.objectid import ObjectId
import dateutil.parser
import datetime
import matplotlib.dates as mdates

from raw_rnn import Dataset

LOG_DIR = './ops_logs/train'
TIMESTEPS = 10
RNN_LAYERS = [{'num_units': 5}]
model_params = {"learning_rate": 0.005, "lstm_dim": 50, "optimizer": "SGD", "steps": [{'num_units': 5}]}
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 5000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

def load_data(filename):
    df = pd.read_csv(filename, names=["date", "in_id", "out_id", "calls", "duration"], parse_dates=['date'])
    df = df['calls'].groupby(df.date).sum()
    df = df.reset_index()
    df = df.set_index("date")

    return df

def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(steps["num_units"], state_is_tuple=True) for steps in params["steps"]]
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
    x_ = tf.unpack(features, axis=1)
    output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
    output_layer = tf.contrib.layers.linear(features, 1)
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"calls": predictions}

    loss = tf.losses.mean_squared_error(predictions, targets)
    #prediction, loss = tflearn.models.linear_regression(output, y)
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(), optimizer=params["optimizer"],
        learning_rate=params["learning_rate"])

    return predictions_dict, loss, train_op

if __name__ == "__main__":

    dataset = Dataset("data/SET1V_01_bs2.csv", "data/SET1V_02_bs2.csv")

    X_train, Y_train = dataset.train.features, dataset.train.targets
    X_test, Y_test = dataset.test.features, dataset.test.targets
    nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
    nn.fit(x=X_train, y=Y_train, steps=5000)
    ev = nn.evaluate(x=X_test, y=Y_test, steps=1)
    loss_score = ev["loss"]
    print("Loss: %s" % loss_score)
