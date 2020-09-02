import Data_Process, Model
import tensorflow as tf
import pickle, sys, os
from multiprocessing import Process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def main(inputs):
    # For select model and activation function
    mod_dict = {'c':'cnn', 'l':'lstm', 'b':'bilstm', 't':'tcn'}
    act_dict = {'s': 'sigmoid', 't': 'tanh', 'r': 'relu'}

    size, init_lr, select, start_bar, activations, modes = inputs
    float_init_lr = 10 ** (-int(init_lr))

    log_path = 'logs/'
    os.makedirs(log_path, exist_ok=True)

    try:
        with open('data/size' + size + '.pkl', 'rb') as file:
            tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = pickle.load(file)
    except:
        tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = Data_Process.main(int(size))
        with open('data/size' + size + '.pkl', 'wb') as file:
            pickle.dump((tickers, train_date, valid_date, test_date, train_data, valid_data, test_data), file,
                        protocol=4)

    # Dataset select the return label and normalization
    train_data = Data_Process.dataset_normalize(train_data, select, start_bar)
    valid_data = Data_Process.dataset_normalize(valid_data, select, start_bar)
    test_data = Data_Process.dataset_normalize(test_data, select, start_bar)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('GPU working', tf.test.is_gpu_available())

    input_shape = train_data[0].shape[1:]
    batch_size = 10000
    train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000000).batch(batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size)
    test_data_tf = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)

    for mod in modes:
        for act in activations:

            mode = mod_dict[mod]
            activation = act_dict[act]
            log_path = 'logs/' + '_'.join([size, init_lr, select, start_bar]) + '/' + act + '/'
            os.makedirs(log_path, exist_ok=True)

            if mode == 'cnn':
                model = Model.CNN_Pred(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                       num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                                       kernel_size=(2,1), pool_size=(2,1), strides=(2,1),
                                       activation=activation)
            elif mode == 'tcn':
                model = Model.TCN_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                        num_dense=16, activation=activation)

            else:
                model = Model.LSTM_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                                         num_dense=16, activation=activation)

            model.load(log_path + mode + '_model.h5')
            model.summary()

            history = model.fit(train_data, valid_data, epochs=50, filepath=log_path)
            test_loss, test_metrics = model.evaluate(test_data_tf)
            print('Test Loss', test_loss, 'Test Metrics', test_metrics)

            # test_LOSS, test_IC = get_perform(model, test_data)
            # print('Self test!')
            # print('Test Loss', test_LOSS, 'Test Metrics', test_IC)

            plot_history(history, test_loss, test_metrics, mode, log_path)
            with open(log_path + mode + '_history.pkl', 'wb') as file:
                pickle.dump((history.history, test_loss, test_metrics, test_loss, test_metrics), file)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[7]
    main(sys.argv[1:-1])

