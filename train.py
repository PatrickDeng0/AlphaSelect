import Data_Process, Model
import tensorflow as tf
import pickle, sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


def plot_history(history, test_loss, test_metrics, mode, log_path):
    try:
        history_df = pd.DataFrame(history.history)
    except:
        history_df = pd.DataFrame(history)

    # fig1: Loss function and IC changes during training
    fig1 = plt.figure(figsize=(18,12))
    ax1 = fig1.add_subplot(231)
    ax1.plot(history_df['loss'], label='train loss')
    ax1.set_title('Train Loss')
    ax1.legend()

    ax2 = fig1.add_subplot(232)
    ax2.plot(history_df['val_loss'], label='valid loss')
    ax2.set_title('Valid Loss')
    ax2.legend()

    ax3 = fig1.add_subplot(233)
    ax3.semilogy(history_df['lr'], label='lr')
    ax3.set_title('Learning Rate')
    ax3.legend()

    ax4 = fig1.add_subplot(234)
    ax4.plot(history_df['IC'], label='train IC')
    ax4.set_title('Train IC')
    ax4.set_ylim(bottom=0, top=0.5)
    ax4.legend()

    ax5 = fig1.add_subplot(235)
    ax5.plot(history_df['val_IC'], label='valid IC')
    ax5.set_title('Valid IC')
    ax5.set_ylim(bottom=0, top=0.5)
    ax5.legend()

    ax6 = fig1.add_subplot(236)
    ax6.annotate('Train Loss %12f' % history_df['loss'].values[-1], xy=(3, 11), xytext=(3, 11), size=12)
    ax6.annotate('Train IC %5f' % history_df['IC'].values[-1], xy=(3, 9), xytext=(3, 9), size=12)
    ax6.annotate('Valid Loss %12f' % history_df['val_loss'].values[-1], xy=(3, 7), xytext=(3, 7), size=12)
    ax6.annotate('Valid IC %5f' % history_df['val_IC'].values[-1], xy=(3, 5), xytext=(3, 5), size=12)
    ax6.annotate('Test Loss %12f' % test_loss, xy=(3, 3), xytext=(3, 3), size=12)
    ax6.annotate('Test IC %5f' % test_metrics, xy=(3, 1), xytext=(3, 1), size=12)
    ax6.set_ylim(bottom=0, top=12)
    ax6.set_xlim(left=0, right=10)
    ax6.set_title('Performance')

    fig1.savefig(log_path + mode + '_perform.jpeg')


def get_perform(model, test_data):
    y_pred = model.predict(test_data[0]).reshape(-1)
    y_true = test_data[1]
    loss = np.mean((y_true - y_pred)**2)
    IC = np.corrcoef(y_true, y_pred)[0, 1]
    return loss, IC


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


def get_signal(model, train_data_signal, valid_data_signal, test_data_signal, log_path, mode):
    def get_signal_perform(model, data_signal):
        signal_X, y_true = data_signal
        y_pred = model.predict(data_signal).reshape(y_true.shape)
        res, IC, w_IC = get_perform_period(y_pred, y_true)
        return res, IC, w_IC

    train_res, train_IC, train_w_IC = get_signal_perform(model, train_data_signal)
    valid_res, valid_IC, valid_w_IC = get_signal_perform(model, valid_data_signal)
    test_res, test_IC, test_w_IC = get_signal_perform(model, test_data_signal)

    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(131)
    for i in range(len(train_res)):
        record = train_res[i]
        ax1.plot(record, label='group '+str(i))
    ax1.set_title('Train IC=%4f, All IC=%4f' % (train_IC, train_w_IC))
    ax1.legend()

    ax2 = fig.add_subplot(132)
    for i in range(len(valid_res)):
        record = valid_res[i]
        ax2.plot(record, label='group '+str(i))
    ax2.set_title('Valid IC=%4f, All IC=%4f' % (valid_IC, valid_w_IC))
    ax2.legend()

    ax3 = fig.add_subplot(133)
    for i in range(len(test_res)):
        record = test_res[i]
        ax3.plot(record, label='group '+str(i))
    ax3.set_title('Test IC=%4f, All IC=%4f' % (test_IC, test_w_IC))
    ax3.legend()

    fig.savefig(log_path + mode + '_pnl.jpeg')


def main(inputs):
    # For select model and activation function
    mod_dict = {'c':'cnn', 'l':'lstm', 'b':'bilstm', 't':'tcn', 'x':'x', 'y':'y'}
    act_dict = {'s': 'sigmoid', 't': 'tanh', 'r': 'relu'}

    size, select, start_bar, markets, activations, modes = inputs
    float_init_lr = 10 ** (-3)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('GPU working', tf.test.is_gpu_available())

    log_path = 'logs/'
    os.makedirs(log_path, exist_ok=True)

    for market in markets:
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

        for mod in modes:
            for act in activations:
                mode = mod_dict[mod]
                activation = act_dict[act]

                print('=============================================================================')
                print('=============================================================================')
                print('Model: %s, Activation: %s' % (mode, activation))
                print(dt.datetime.now())

                log_path = 'logs/' + '_'.join([size, select, start_bar, market]) + '/' + act + '/'
                os.makedirs(log_path, exist_ok=True)

                if mode == 'cnn':
                    model = Model.CNN_Pred(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                           num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                                           kernel_size=(2,1), pool_size=(2,1), strides=(2,1),
                                           activation=activation)

                elif mode == 'tcn':
                    model = Model.TCN_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                            num_dense=16, activation=activation)

                elif mode == 'x':
                    model = Model.X_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                          num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                                          kernel_size=(2,1), pool_size=(2,1), strides=(2,1),
                                          activation=activation)

                elif mode == 'y':
                    model = Model.Y_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                          num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                                          kernel_size=(2,1), pool_size=(2,1), strides=(2,1),
                                          activation=activation)

                else:
                    model = Model.LSTM_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                             num_dense=16, activation=activation)

                model._build_model()
                model.summary()

                history = model.fit(train_data, valid_data, epochs=50)
                model.save_model(filepath=log_path)
                test_loss, test_metrics = model.evaluate(test_data)
                print('Test Loss', test_loss, 'Test Metrics', test_metrics)

                plot_history(history, test_loss, test_metrics, mode, log_path)
                with open(log_path + mode + '_history.pkl', 'wb') as file:
                    pickle.dump((history.history, test_loss, test_metrics), file)

                get_signal(model, train_data_signal, valid_data_signal, test_data_signal, log_path, mode)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[7]
    main(sys.argv[1:-1])
