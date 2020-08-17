import tensorflow as tf
import tensorflow.keras as tk

import numpy as np
import os


class CNN_Pred:
    def __init__(self, input_shape, learning_rate=0.001, num_channel=64, num_hidden=8,
                 pool_method='max', binary='binary'):
        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._num_channel = num_channel
        self._num_hidden = num_hidden
        self._pool_method = pool_method
        self._binary = binary
        self._model = self._build_model()

    def pooling_choice(self, shape):
        if self._pool_method == 'max':
            return tk.layers.MaxPool2D(shape)
        else:
            return tk.layers.AvgPool2D(shape)

    def _build_model(self):
        model = tf.keras.Sequential()
        # Add a channel information
        re_shape = tuple(list(self._input_shape) + [1])
        model.add(tk.layers.Reshape(re_shape, input_shape=self._input_shape))
        model.add(tk.layers.Conv2D(self._num_channel, (1, self._input_shape[1]), activation='relu'))
        model.add(tk.layers.Conv2D(self._num_channel, (3, 1), activation='relu'))
        model.add(self.pooling_choice(shape=(2, 1)))
        model.add(tk.layers.Conv2D(self._num_channel, (3, 1), activation='relu'))
        model.add(self.pooling_choice(shape=(2, 1)))
        model.add(tk.layers.Flatten())
        model.add(tk.layers.Dense(self._num_hidden, activation='relu'))
        model.add(tk.layers.Dense(self._num_hidden, activation='relu'))
        if self._binary == 'binary':
            model.add(tk.layers.Dense(2, activation='softmax'))
            model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                          loss=tk.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        else:
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                          loss=tk.losses.MeanSquaredError(), metrics=['mse'])

        return model

    def summary(self):
        self._model.summary()

    def change_LR(self, learningrate):
        if self._binary == 'binary':
            self._model.compile(optimizer=tk.optimizers.Adam(learning_rate=learningrate),
                                loss=tk.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        else:
            self._model.compile(optimizer=tk.optimizers.Adam(learning_rate=learningrate),
                                loss=tk.losses.MeanSquaredError(), metrics=['mse'])

    def fit(self, train_data, valid_data, epochs, filepath):
        if self._binary == 'binary':
            es = tk.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', patience=20, verbose=2)
            low_LR = tk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, mode='auto',
                                                    min_lr=0.000001, verbose=2)
        else:
            es = tk.callbacks.EarlyStopping(monitor='val_mse', mode='auto', patience=20, verbose=2)
            low_LR = tk.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=10, mode='auto',
                                                    min_lr=0.000001, verbose=2)

        self._model.fit(train_data, validation_data=valid_data, epochs=epochs, callbacks=[es, low_LR])
        self._model.save(filepath + 'model.h5')

    def evaluate(self, test_data):
        return self._model.evaluate(test_data)

    def predict(self, test_X):
        return self._model.predict(test_X)

    def load(self, filename):
        try:
            self._model = tf.keras.models.load_model(filename)
            print('Load model successful!')
        except:
            print('Create New Model now!')
