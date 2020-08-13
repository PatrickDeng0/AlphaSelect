import util, Data_Process, Model
import tensorflow as tf
import datetime as dt
import sys, os


size = sys.argv[1]
binary = sys.argv[2]
mini = sys.argv[3]
pooling = sys.argv[4]
log_path = '_'.join(sys.argv[1:]) + '/'
os.makedirs(log_path, exist_ok=True)

batch_size = 512
size = int(size)
tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = Data_Process.main(size)

if mini == 'mini':
    train_data = (train_data[0][-100000:], train_data[1][-100000:])
    valid_data = (valid_data[0][-10000:], valid_data[1][-10000:])
    test_data = (test_data[0][-10000:], test_data[1][-10000:])

if binary == 'bin':
    train_data = Data_Process.binarize_data(train_data)
    valid_data = Data_Process.binarize_data(valid_data)
    test_data = Data_Process.binarize_data(test_data)

input_shape = train_data[0].shape[1:]
train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
valid_data = tf.data.Dataset.from_tensor_slices(valid_data).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)


model = Model.CNN_Pred(input_shape=input_shape, learning_rate=0.001, num_channel=8,
                       num_hidden=64, pool_method=pooling, binary=binary)
model.summary()
model.fit(train_data, valid_data, epochs=10, filepath=log_path)
loss, metrics = model.evaluate(test_data)
print('Test Loss', loss)
print('Test Metrics', metrics)
