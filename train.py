import Data_Process, Model
import tensorflow as tf
import pickle, sys, os


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

    with open(log_path + mode + '_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    print('LR after training:', model._model.optimizer.lr.numpy())
    loss, metrics = model.evaluate(test_data)
    print('Test Loss', loss, 'Test Metrics', metrics)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[5]
    main(sys.argv[1:-1])
