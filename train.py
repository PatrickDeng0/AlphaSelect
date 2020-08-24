import Data_Process, Model
import tensorflow as tf
import pickle, sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, test_loss, test_metrics, mode, log_path):
    try:
        history_df = pd.DataFrame(history.history)
    except:
        history_df = pd.DataFrame(history)

    # fig1: Loss function and IC changes during training
    fig1 = plt.figure(1, figsize=(18,12))
    ax1 = fig1.add_subplot(231)
    ax1.plot(history_df['loss'], label='train loss')
    ax1.set_title('Train Loss')
    ax1.legend()

    ax2 = fig1.add_subplot(232)
    ax2.plot(history_df['val_loss'], label='valid loss')
    ax2.set_title('Valid Loss')
    ax2.legend()

    ax3 = fig1.add_subplot(233)
    ax3.semilogy(history_df['lr'])
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
    ax6.annotate('Test Loss %12f' % test_loss, xy=(3, 4), xytext=(3, 4), size=18)
    ax6.annotate('Test IC %5f' % test_metrics, xy=(3, 6), xytext=(3, 6), size=18)
    ax6.set_ylim(bottom=0, top=10)
    ax6.set_xlim(left=0, right=10)
    ax6.set_title('Test Performance')

    fig1.savefig(log_path + mode + '_perform.jpeg')


def get_perform(model, test_data):
    y_pred = model.predict(test_data[0])
    y_true = test_data[1]
    loss = np.mean((y_true - y_pred)**2)
    IC = np.corrcoef(y_true, y_pred)[0, 1]
    return loss, IC


# read model and get its perform
def perform(size, batch_size, init_lr, mode):
    log_path = 'logs/' + '_'.join([size, batch_size, init_lr]) + '/'
    try:
        with open('data/size' + size + '.pkl', 'rb') as file:
            tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = pickle.load(file)
    except:
        print('Data not prepared!')
        return

    test_data = Data_Process.dataset_normalize(test_data)
    input_shape = test_data[0].shape[1:]
    if mode == 'cnn':
        model = Model.CNN_Pred(input_shape=input_shape, learning_rate=init_lr, num_channel=64, num_hidden=16,
                               kernel_size=(3, 1), pool_size=(2, 1))
        model.load(log_path + 'CNN_model.h5')

    else:
        model = Model.LSTM_model(input_shape=input_shape, learning_rate=init_lr, num_hidden=16)
        model.load(log_path + 'LSTM_model.h5')

    with open(log_path + mode + '_history.pkl', 'wb') as file:
        content = pickle.load(file)
    if isinstance(content, tuple):
        history = content[0]
    else:
        history = content

    test_loss, test_metrics = get_perform(model, test_data)
    plot_history(history, test_loss, test_metrics, mode, log_path)


def main(inputs):
    size, batch_size, init_lr, mode = inputs
    log_path = 'logs/' + '_'.join(inputs[:-1]) + '/'
    batch_size = int(batch_size)
    init_lr = 10 ** (-int(init_lr))
    try:
        with open('data/size' + size + '.pkl', 'rb') as file:
            tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = pickle.load(file)
    except:
        tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = Data_Process.main(int(size))
        with open('data/size' + size + '.pkl', 'wb') as file:
            pickle.dump((tickers, train_date, valid_date, test_date, train_data, valid_data, test_data), file,
                        protocol=4)

    train_data = Data_Process.dataset_normalize(train_data)
    valid_data = Data_Process.dataset_normalize(valid_data)
    test_data = Data_Process.dataset_normalize(test_data)

    print('GPU working', tf.test.is_gpu_available())

    input_shape = train_data[0].shape[1:]
    train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000000).batch(batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)

    if mode == 'cnn':
        model = Model.CNN_Pred(input_shape=input_shape, learning_rate=init_lr, num_channel=64, num_hidden=16,
                               kernel_size=(3, 1), pool_size=(2, 1))
        model.summary()
        model.load(log_path + 'CNN_model.h5')
    else:
        model = Model.LSTM_model(input_shape=input_shape, learning_rate=init_lr, num_hidden=16)
        model.summary()
        model.load(log_path + 'LSTM_model.h5')

    print('LR before training:', model._model.optimizer.lr.numpy())
    history = model.fit(train_data, valid_data, epochs=100, filepath=log_path)

    print('LR after training:', model._model.optimizer.lr.numpy())
    test_loss, test_metrics = model.evaluate(test_data)
    print('Test Loss', test_loss, 'Test Metrics', test_metrics)

    plot_history(history, test_loss, test_metrics, mode, log_path)
    with open(log_path + mode + '_history.pkl', 'wb') as file:
        pickle.dump((history.history, test_loss, test_metrics), file)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[5]
    main(sys.argv[1:-1])
