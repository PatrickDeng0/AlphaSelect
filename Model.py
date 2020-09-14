import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.backend as K
from tcn import TCN, tcn_full_summary


def IC(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.abs(K.mean(r))


class Model:
    def __init__(self, mode, learning_rate, activation):
        self._mode = mode
        self._learning_rate = learning_rate
        self._activation = activation
        self._model = None

    def _build_model(self):
        # Customized build model!
        pass

    def summary(self):
        if self._mode == 'tcn':
            tcn_full_summary(self._model)
        else:
            self._model.summary()

    def change_LR(self, learningrate):
        self._learning_rate = learningrate
        self._model.compile(optimizer=tk.optimizers.Adam(learning_rate=learningrate),
                            loss=tk.losses.MeanSquaredError(), metrics=[IC])

    def fit(self, train_data, valid_data, epochs, filepath):
        rlreduce = tk.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                                  mode='auto', cooldown=5, min_lr=10 ** (-6))
        ES = tk.callbacks.EarlyStopping(monitor='val_loss', patience=40, mode='min')
        history = self._model.fit(train_data, validation_data=valid_data, epochs=epochs,
                                  callbacks=[rlreduce, ES], verbose=2)
        self._model.save(filepath + self._mode + '_model.h5')
        return history

    def evaluate(self, test_data):
        return self._model.evaluate(test_data)

    def predict(self, test_X):
        return self._model.predict(test_X)

    def load(self, filename):
        try:
            self._model = tf.keras.models.load_model(filename, custom_objects={'IC': IC})
            print('Load Succeed')
        except:
            self._build_model()
            print('New Model Create')


class CNN_Pred(Model):
    def __init__(self, mode, input_shape, learning_rate=0.001, num_vr_kernel=64, num_time_kernel=16, num_dense=16,
                 kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1), activation='relu'):
        super().__init__(mode, learning_rate, activation)
        self._input_shape = input_shape
        self._num_vr_kernel = num_vr_kernel
        self._num_time_kernel = num_time_kernel
        self._num_dense = num_dense

        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._strides = strides
        self._conv_layer = self._get_conv_layer()

    def _get_conv_layer(self):
        if self._input_shape[0] % 16 == 0:
            days = self._input_shape[0] // 16
        else:
            days = self._input_shape[0] // 16 + 1
        res = 0
        while days > 4:
            days = days // 2 + 1
            res += 1
        return res

    def _build_model(self):
        inputs = tk.layers.Input(shape=self._input_shape)
        re_shape = tuple(list(self._input_shape) + [1])
        x = tk.layers.Reshape(re_shape, input_shape=self._input_shape)(inputs)
        x = tk.layers.Conv2D(self._num_vr_kernel, (1, self._input_shape[1]), activation='relu')(x)

        for _ in range(2):
            x = tk.layers.Conv2D(self._num_time_kernel, self._kernel_size, strides=self._strides,
                                 activation='relu', padding='same')(x)
            max_pool = tk.layers.MaxPool2D(self._pool_size, padding='same')(x)
            aver_pool = tk.layers.AvgPool2D(self._pool_size, padding='same')(x)
            x = tk.layers.Concatenate(axis=3)([max_pool, aver_pool])

        x = tk.layers.Dropout(rate=0.5)(x)
        for _ in range(self._conv_layer):
            x = tk.layers.Conv2D(self._num_time_kernel*2, self._kernel_size, strides=self._strides,
                                 activation='relu', padding='same')(x)

        x = tk.layers.Flatten()(x)
        x = tk.layers.Dropout(rate=0.5)(x)
        x = tk.layers.Dense(self._num_dense, activation=self._activation)(x)
        output = tk.layers.Dense(1)(x)
        model = tk.Model(inputs=inputs, outputs=output)

        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model


class LSTM_Model(Model):
    def __init__(self, mode, input_shape, learning_rate=0.001, num_dense=16, activation='relu'):
        super().__init__(mode, learning_rate, activation)
        self._input_shape = input_shape
        self._num_dense = num_dense

    def _build_model(self):
        model = tk.Sequential()
        if self._mode == 'bilstm':
            model.add(tk.layers.Bidirectional(tk.layers.LSTM(units=self._num_dense),
                                              input_shape=self._input_shape))
        else:
            model.add(tk.layers.LSTM(units=self._num_dense, input_shape=self._input_shape))
        model.add(tk.layers.Dense(units=self._num_dense, activation=self._activation))
        model.add(tk.layers.Dense(units=1))
        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model


class TCN_Model(Model):
    def __init__(self, mode, input_shape, learning_rate=0.001, num_dense=16, activation='relu'):
        super().__init__(mode, learning_rate, activation)
        self._input_shape = input_shape
        self._num_dense = num_dense

    def _build_model(self):
        inputs = tk.layers.Input(shape=self._input_shape)

        # TCN layer
        x = TCN(return_sequences=False, nb_filters=self._num_dense, activation='relu')(inputs)
        x = tk.layers.Dense(self._num_dense, activation=self._activation)(x)
        output = tk.layers.Dense(1)(x)
        model = tk.Model(inputs=inputs, outputs=output)

        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model


class X_Model(Model):
    def __init__(self, mode, input_shape, learning_rate=0.001, num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                 kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1), activation='relu'):
        super().__init__(mode, learning_rate, activation)
        self._input_shape = input_shape
        self._num_vr_kernel = num_vr_kernel
        self._num_time_kernel = num_time_kernel
        self._num_dense = num_dense

        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._strides = strides
        self._conv_layer = 2

    def _build_model(self):
        inputs = tk.layers.Input(shape=self._input_shape)
        re_shape = tuple(list(self._input_shape) + [1])
        x = tk.layers.Reshape(re_shape, input_shape=self._input_shape)(inputs)
        x = tk.layers.Conv2D(self._num_vr_kernel, (1, self._input_shape[1]), activation='relu')(x)

        for _ in range(self._conv_layer):
            x = tk.layers.Conv2D(self._num_time_kernel, self._kernel_size, strides=self._strides,
                                 activation='relu')(x)
            max_pool = tk.layers.MaxPool2D(self._pool_size)(x)
            aver_pool = tk.layers.AvgPool2D(self._pool_size)(x)
            x = tk.layers.Concatenate(axis=3)([max_pool, aver_pool])

        x = tk.backend.squeeze(x, axis=2)
        x = tk.layers.LSTM(units=self._num_dense)(x)
        x = tk.layers.Dense(units=self._num_dense, activation=self._activation)(x)
        output = tk.layers.Dense(1)(x)

        model = tk.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model


class Y_Model(Model):
    def __init__(self, mode, input_shape, learning_rate=0.001, num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                 kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1), activation='relu'):
        super().__init__(mode, learning_rate, activation)
        self._input_shape = input_shape
        self._num_vr_kernel = num_vr_kernel
        self._num_time_kernel = num_time_kernel
        self._num_dense = num_dense

        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._strides = strides
        self._conv_layer = 2

        self._bar = self._input_shape[0] % 16
        self._part1_shape = (self._input_shape[0] - self._bar, self._input_shape[1])
        self._part2_shape = (self._bar, self._input_shape[1])

    def _build_model(self):
        inputs = tk.layers.Input(shape=self._input_shape)
        re_shape = tuple(list(self._input_shape) + [1])
        x = tk.layers.Reshape(re_shape, input_shape=self._input_shape)(inputs)
        x = tk.layers.Conv2D(self._num_vr_kernel, (1, self._input_shape[1]), activation='relu')(x)

        # Divided into 2 parts of input
        # x1: Daily frequent data
        # x2: Intraday frequent data
        x1 = tf.slice(x, [0, 0, 0, 0], [-1, self._part1_shape[0], -1, -1])
        x2 = tf.slice(x, [0, self._part1_shape[0], 0, 0], [-1, -1, -1, -1])

        # For input1 model: 2 conv-pool, then squeeze and lstm
        for _ in range(self._conv_layer):
            x1 = tk.layers.Conv2D(self._num_time_kernel, self._kernel_size, strides=self._strides,
                                  activation='relu')(x1)
            max_pool = tk.layers.MaxPool2D(self._pool_size)(x1)
            aver_pool = tk.layers.AvgPool2D(self._pool_size)(x1)
            x1 = tk.layers.Concatenate(axis=3)([max_pool, aver_pool])
        x1 = tk.backend.squeeze(x1, axis=2)
        _, x1, x1_state = tk.layers.LSTM(units=self._num_dense, return_state=True)(x1)

        x2 = tk.backend.squeeze(x2, axis=2)
        _, _, pred = tk.layers.LSTM(units=self._num_dense, return_state=True)(x2, initial_state=[x1, x1_state])
        pred = tk.layers.Dense(units=self._num_dense, activation=self._activation)(pred)
        output = tk.layers.Dense(1)(pred)

        model = tk.Model(inputs=inputs, outputs=output)
        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model
