import Data_Process, Model
import tensorflow as tf
import pickle, sys, os
import pandas as pd
import numpy as np
import datetime as dt


def divide_bins_average(y_true_slice, ranks):
    num_each_bins = len(ranks) // 10
    res = []
    for i in range(10):
        if i < 9:
            stocks = ranks[i*num_each_bins: (i+1)*num_each_bins]
        else:
            stocks = ranks[i*num_each_bins:]
        select = y_true_slice[stocks]
        bin_aver = np.nanmean(select)
        res.append(bin_aver)
    return np.array(res)


def get_perform_period(y_pred, y_true):
    res, IC, pred_cum, true_cum = [], [], np.array([]), np.array([])
    for i in range(y_pred.shape[0]):
        y_pred_slice, y_true_slice = y_pred[i], y_true[i]
        valid_stocks = np.where((~np.isnan(y_pred_slice)) & (~np.isnan(y_true_slice)))[0]
        y_pred_slice, y_true_slice = y_pred_slice[valid_stocks], y_true_slice[valid_stocks]

        pred_cum = np.concatenate([pred_cum, y_pred_slice], axis=0)
        true_cum = np.concatenate([true_cum, y_true_slice], axis=0)

        IC.append(np.corrcoef(y_pred_slice, y_true_slice)[0,1])
        pred_rank = np.argsort(y_pred_slice)
        bins_average = divide_bins_average(y_true_slice, pred_rank)
        res.append(bins_average)
    res = np.array(res).T
    res = res + 1
    res = np.cumprod(res, axis=1)
    return res, np.nanmean(np.array(IC)), np.corrcoef(pred_cum, true_cum)[0,1]


def get_signal_perform(model, data_signal):
    signal_X, y_true = data_signal
    y_pred = model.predict(data_signal).reshape(y_true.shape)
    res, IC, w_IC = get_perform_period(y_pred, y_true)
    return res, IC, w_IC


def main(inputs):
    # For select model and activation function
    mod_dict = {'c':'cnn', 'l':'lstm', 'b':'bilstm', 't':'tcn', 'x':'x', 'y':'y'}
    act_dict = {'s': 'sigmoid', 't': 'tanh', 'r': 'relu'}
    opt_dict = {'r':'RMSprop', 'n':'Nadam', 'a':'Adam'}

    size, select, start_bar, market, activation, mode, optimizer = inputs
    mode = mod_dict[mode]
    activation = act_dict[activation]
    optimizer = opt_dict[optimizer]
    float_init_lr = 10 ** (-3)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('GPU working', tf.test.is_gpu_available())

    log_path = 'logs_2/'
    os.makedirs(log_path, exist_ok=True)

    tickers, train_date, valid_date, test_date, train_data, valid_data, test_data \
        = Data_Process.main(int(size), int(select), int(start_bar), market)

    train_data, train_data_signal = train_data
    valid_data, valid_data_signal = valid_data
    test_data, test_data_signal = test_data

    print('Train dates:', train_date[0], train_date[-1])
    print('Valid dates:', valid_date[0], valid_date[-1])
    print('Test dates:', test_date[0], test_date[-1])
    print('=============================================================================')

    input_shape = train_data[0].shape[1:]
    batch_size = 10000
    train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000000).batch(batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)

    print('=============================================================================')
    print('=============================================================================')
    print('Model: %s, Activation: %s, Optimizer: %s' % (mode, activation, optimizer))
    print(dt.datetime.now())

    log_path = 'logs_2/' + '_'.join([size, select, start_bar, market]) + '/' + optimizer + '/'
    os.makedirs(log_path, exist_ok=True)

    batch_IC_res, signal_IC_res = [], []
    for i in range(10):
        print('=============================================================================')
        print('=============================================================================')
        print('Validation %d' % i)

        model = Model.model_select(mode, input_shape, float_init_lr, activation, optimizer)
        model._build_model()
        model.summary()
        history = model.fit(train_data, valid_data, epochs=50)

        test_loss, test_metrics = model.evaluate(test_data)
        print('Test Loss', test_loss, 'Test Metrics', test_metrics)

        history_df = pd.DataFrame(history.history)
        batch_IC_res.append([history_df['IC'].values[-1], history_df['val_IC'].values[-1], test_metrics])

        _, signal_train_IC, _ = get_signal_perform(model, train_data_signal)
        _, signal_valid_IC, _ = get_signal_perform(model, valid_data_signal)
        _, signal_test_IC, _ = get_signal_perform(model, test_data_signal)

        signal_IC_res.append([signal_train_IC, signal_valid_IC, signal_test_IC])

    batch_IC_res, signal_IC_res = np.array(batch_IC_res), np.array(signal_IC_res)

    print('Batch IC mean:', batch_IC_res.mean(axis=0))
    print('Signal IC mean:', signal_IC_res.mean(axis=0))

    with open(log_path + mode + '_ICres.pkl', 'wb') as file:
        pickle.dump((batch_IC_res, signal_IC_res), file)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[8]
    main(sys.argv[1:-1])
