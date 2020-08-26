import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.backend as K


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


class CNN_Pred:
    def __init__(self, input_shape, learning_rate=0.001,
                 num_vr_kernel=64, num_time_kernel=16, num_dense=16,
                 kernel_size=(3,1), pool_size=(2,1)):

        self._input_shape = input_shape
        self._learning_rate = learning_rate

        self._num_vr_kernel = num_vr_kernel
        self._num_time_kernel = num_time_kernel
        self._num_dense = num_dense

        self._kernel_size = kernel_size
        self._pool_size = pool_size
        self._conv_layer = self._get_conv_layer()

        self._model = None

    def _get_conv_layer(self):
        time_shape, ks, ps = self._input_shape[0], self._kernel_size[0]-1, self._pool_size[0]
        res = 0
        while True:
            time_shape = (time_shape - ks) // ps
            print(time_shape)
            res += 1
            if time_shape < 8:
                break
        return res

    def _build_model(self):
        inputs = tk.layers.Input(shape=self._input_shape)
        re_shape = tuple(list(self._input_shape) + [1])
        x = tk.layers.Reshape(re_shape, input_shape=self._input_shape)(inputs)
        x = tk.layers.Conv2D(self._num_vr_kernel, (1, self._input_shape[1]), activation='relu')(x)

        for _ in range(self._conv_layer):
            x = tk.layers.Conv2D(self._num_time_kernel, self._kernel_size, activation='relu')(x)
            max_pool = tk.layers.MaxPool2D(self._pool_size)(x)
            aver_pool = tk.layers.AvgPool2D(self._pool_size)(x)
            x = tk.layers.Concatenate(axis=3)([max_pool, aver_pool])

        x = tk.layers.Flatten()(x)
        x = tk.layers.Dense(self._num_dense, activation='relu')(x)
        x = tk.layers.Dense(self._num_dense, activation='relu')(x)

        output = tk.layers.Dense(1)(x)
        model = tk.Model(inputs=inputs, outputs=output)

        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model

    def summary(self):
        self._model.summary()

    def change_LR(self, learningrate):
        self._model.compile(optimizer=tk.optimizers.Adam(learning_rate=learningrate),
                            loss=tk.losses.MeanSquaredError(), metrics=[IC])

    def fit(self, train_data, valid_data, epochs, filepath):
        rlreduce = tk.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                                  mode='auto', cooldown=5, min_lr=10**(-6))
        ES = tk.callbacks.EarlyStopping(monitor='val_loss', patience=40, mode='min')
        history = self._model.fit(train_data, validation_data=valid_data, epochs=epochs, callbacks=[rlreduce, ES])
        self._model.save(filepath + 'CNN_model.h5')
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


class LSTM_model:
    def __init__(self, input_shape, learning_rate=0.001, num_dense=16):
        self._input_shape = input_shape
        self._learning_rate = learning_rate
        self._num_dense = num_dense
        self._model = None

    def _build_model(self):
        model = tk.Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self._num_dense),
                                                input_shape=self._input_shape))
        model.add(tf.keras.layers.Dense(units=self._num_dense, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(units=1))
        model.compile(optimizer=tk.optimizers.Adam(learning_rate=self._learning_rate),
                      loss=tk.losses.MeanSquaredError(), metrics=[IC])
        self._model = model

    def summary(self):
        self._model.summary()

    def change_LR(self, learningrate):
        self._model.compile(optimizer=tk.optimizers.Adam(learning_rate=learningrate),
                            loss=tk.losses.MeanSquaredError(), metrics=[IC])

    def fit(self, train_data, valid_data, epochs, filepath):
        rlreduce = tk.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                                  mode='auto', cooldown=5, min_lr=10 ** (-6))
        ES = tk.callbacks.EarlyStopping(monitor='val_loss', patience=40, mode='min')
        history = self._model.fit(train_data, validation_data=valid_data, epochs=epochs, callbacks=[rlreduce, ES])
        self._model.save(filepath + 'LSTM_model.h5')
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
