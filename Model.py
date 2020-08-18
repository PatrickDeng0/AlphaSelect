import tensorflow as tf
import tensorflow.keras as tk
import tensorflow_probability as tfp

import numpy as np
import os


class IC(tf.keras.metrics.Metric):
    def __init__(self, name='IC', **kwargs):
        super(IC, self).__init__(name=name, **kwargs)
        self.value = self.add_weight(name='IC', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        corr = tfp.stats.correlation(y_true, y_pred)
        self.value.assign_add(tf.reduce_mean(tf.square(corr)))

    def result(self):
        return self.value


class CNN_Pred:
    def __init__(self, input_shape, learning_rate=0.001, num_channel=64, num_hidden=16,
                 kernel_size=(3,1), pool_size=(2,1)):
        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._num_channel = num_channel
        self._num_hidden = num_hidden
        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._model = self._build_model()

    def _build_model(self):
        inputs = tk.layers.Input(shape=self._input_shape)
        re_shape = tuple(list(self._input_shape) + [1])
        x = tk.layers.Reshape(re_shape, input_shape=self._input_shape)(inputs)
        x = tk.layers.Conv2D(self._num_channel, (1, self._input_shape[1]), activation='relu')(x)
        x = tk.layers.Conv2D(self._num_channel//2, self._kernel_size, activation='relu')(x)

        max_pool_1 = tk.layers.MaxPool2D(self._pool_size)(x)
        aver_pool_1 = tk.layers.AvgPool2D(self._pool_size)(x)

        x = tk.layers.Concatenate(axis=3)([max_pool_1, aver_pool_1])
        x = tk.layers.Conv2D(self._num_channel//2, self._kernel_size, activation='relu')(x)

        max_pool_2 = tk.layers.MaxPool2D(self._pool_size)(x)
        aver_pool_2 = tk.layers.AvgPool2D(self._pool_size)(x)

        x = tk.layers.Concatenate(axis=3)([max_pool_2, aver_pool_2])
        x = tk.layers.Conv2D(self._num_channel//2, self._kernel_size, activation='relu')(x)

        max_pool_3 = tk.layers.MaxPool2D(self._pool_size)(x)
        aver_pool_3 = tk.layers.AvgPool2D(self._pool_size)(x)

        x = tk.layers.Concatenate(axis=3)([max_pool_3, aver_pool_3])
        x = tk.layers.Flatten()(x)
        x = tk.layers.Dense(self._num_hidden, activation='relu')(x)
        x = tk.layers.Dense(self._num_hidden, activation='relu')(x)

        output = tk.layers.Dense(1)(x)
        model = tk.Model(inputs=inputs, outputs=output)

        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC()])
        return model

    def summary(self):
        self._model.summary()

    def change_LR(self, learningrate):
        self._model.compile(optimizer=tk.optimizers.Adam(learning_rate=learningrate),
                            loss=tk.losses.MeanSquaredError(), metrics=[IC()])

    def fit(self, train_data, valid_data, epochs, filepath):
        es = tk.callbacks.EarlyStopping(monitor='val_IC', mode='max', patience=20, verbose=2)
        low_LR = tk.callbacks.ReduceLROnPlateau(monitor='val_IC', factor=0.1, patience=5, mode='max',
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
