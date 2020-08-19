import util, Data_Process, Model
import tensorflow as tf
import datetime as dt
import sys, os, pickle


size = sys.argv[1]
mini = sys.argv[2]
batch_size = sys.argv[3]
init_lr = sys.argv[4]
mode = sys.argv[5]
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[6]

log_path = 'logs/'
os.makedirs(log_path, exist_ok=True)
log_path = log_path + '_'.join(sys.argv[1:5]) + '/'
os.makedirs(log_path, exist_ok=True)

batch_size = int(batch_size)
init_lr = 10**(-int(init_lr))
try:
    with open('data/size'+size+'.pkl', 'rb') as file:
        tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = pickle.load(file)
except:
    tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = Data_Process.main(int(size))
    with open('data/size' + size + '.pkl', 'wb') as file:
        pickle.dump((tickers, train_date, valid_date, test_date, train_data, valid_data, test_data), file, protocol=4)

train_data = Data_Process.dataset_normalize(train_data)
valid_data = Data_Process.dataset_normalize(valid_data)
test_data = Data_Process.dataset_normalize(test_data)

if mini == 'mini':
    train_data = (train_data[0][:100000], train_data[1][:100000])
    valid_data = (valid_data[0][-10000:], valid_data[1][-10000:])
    test_data = (test_data[0][-10000:], test_data[1][-10000:])

print('GPU working', tf.test.is_gpu_available())

input_shape = train_data[0].shape[1:]
train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000000).batch(batch_size)
valid_data = tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)


if mode == 'cnn':
    model = Model.CNN_Pred(input_shape=input_shape, learning_rate=init_lr, num_channel=64, num_hidden=16,
                           kernel_size=(3,1), pool_size=(2,1))
    model.summary()
    model.load(log_path + 'CNN_model.h5')
else:
    model = Model.LSTM_model(input_shape=input_shape, learning_rate=init_lr, num_hidden=16)
    model.summary()
    model.load(log_path + 'LSTM_model.h5')


print('LR before training:', model._model.optimizer.lr.numpy())
model.fit(train_data, valid_data, epochs=50, filepath=log_path)
print('LR after training:', model._model.optimizer.lr.numpy())
loss, metrics = model.evaluate(test_data)
print('Test Loss', loss, 'Test Metrics', metrics)


# usage: python main.py 4 full 8192 2 lstm 0 > logs/4_full_8192_2/lstm_train.log
