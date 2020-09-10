import Data_Process, Model
import pandas as pd
import numpy as np
import h5py, sys
import datetime as dt
import matplotlib.pyplot as plt


def get_res(model, size, Y_select, bar):
    tickers, dates, dataset, train_split, test_split = Data_Process.extract_2_X()
    daily, features, stock_rets, market_rets = dataset[0], dataset[1], \
                                               dataset[2][Y_select,:,:,bar], dataset[3][Y_select,:,bar]
    st_state, trade_state = daily[0], daily[1]
    num_dates = dates.shape[0]
    num_tickers = tickers.shape[0]

    data = []
    input_shape = (size*16+bar+1, 8)
    for date in range(num_dates-size):
        for t in range(num_tickers):
            st_series = st_state[date:(date+size+1), t]
            daily_part_ele_data = features[:, date:(date+size), t, :].reshape((8, -1))
            intra_part_ele_data = features[:, date+size, t, :(bar+1)].reshape((8, -1))
            ele_data = np.concatenate([daily_part_ele_data, intra_part_ele_data], axis=1)
            if np.sum(st_series) == 0 and trade_state[date+size-1, t] and not np.isnan(np.sum(ele_data)):
                ele_data = Data_Process.ele_normalize(ele_data, full=False)
            else:
                ele_data = np.full(input_shape, np.nan)
            data.append(ele_data)

    data = np.array(data)
    y_pred = model.predict(data).reshape((num_dates-size, num_tickers))
    print(y_pred.shape)
    y_true = stock_rets[size:] - market_rets[size:, np.newaxis]
    print(y_true.shape)
    dates = dates[size:]
    return y_pred, y_true, dates, tickers, train_split, test_split


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


def get_perform_period(y_pred_cut, y_true, indexs):
    res, IC = [], []
    for index in indexs:
        y_pred_slice, y_true_slice = y_pred_cut[index], y_true[index]
        valid_stocks = np.where((~np.isnan(y_pred_slice)) & (~np.isnan(y_true_slice)))[0]
        y_pred_slice, y_true_slice = y_pred_slice[valid_stocks], y_true_slice[valid_stocks]

        IC.append(np.corrcoef(y_pred_slice, y_true_slice)[0,1])
        pred_rank = np.argsort(y_pred_slice)
        bins_average = divide_bins_average(y_true_slice, pred_rank)
        res.append(bins_average)
    res = np.array(res).T
    res = res + 1
    res = np.cumprod(res, axis=1)
    return res, np.nanmean(np.array(IC))


def get_perform(y_pred, y_true, dates, train_split, test_split, model_prefix):
    index_train = np.where(dates < dates[train_split])[0]
    index_valid = np.where((dates < dates[test_split]) & (dates >= dates[train_split]))[0]
    index_test = np.where(dates >= dates[test_split])[0]
    train_res, train_IC = get_perform_period(y_pred, y_true, index_train)
    valid_res, valid_IC = get_perform_period(y_pred, y_true, index_valid)
    test_res, test_IC = get_perform_period(y_pred, y_true, index_test)

    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(131)
    for i in range(len(train_res)):
        record = train_res[i]
        ax1.plot(record, label='group '+str(i))
    ax1.set_title('Train IC=%4f' % train_IC)
    ax1.legend()

    ax2 = fig.add_subplot(132)
    for i in range(len(valid_res)):
        record = valid_res[i]
        ax2.plot(record, label='group '+str(i))
    ax2.set_title('Valid IC=%4f' % valid_IC)
    ax2.legend()

    ax3 = fig.add_subplot(133)
    for i in range(len(test_res)):
        record = test_res[i]
        ax3.plot(record, label='group '+str(i))
    ax3.set_title('Test IC=%4f' % test_IC)
    ax3.legend()

    fig.savefig('models/' + model_prefix + '_pnl.jpeg')


def main(size, Y_select, bar):
    input_shape = (size*16+bar+1, 8)
    mode, float_init_lr, activation = 'x', 0.0001, 'relu'

    model_prefix = mode + '_' + str(size) + '_' + str(Y_select) + '_' + str(bar)
    model = Model.X_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                          num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                          kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1),
                          activation=activation)
    model.load('models/' + model_prefix + '_model.h5')
    y_pred, y_true, dates, tickers, train_split, test_split = get_res(model, size, Y_select, bar)

    get_perform(y_pred, y_true, dates, train_split, test_split, model_prefix)

    tickers_t = []
    for ticker in tickers:
        tickers_t.append(ticker.encode())

    with h5py.File('models/' + model_prefix + '_signal.h5', 'w') as h5f:
        h5f.create_dataset('signals', data=y_pred)
        h5f.create_dataset('y_true', data=y_true)
        h5f.create_dataset('dates', data=dates)
        h5f.create_dataset('tickers', data=tickers_t)


if __name__ == '__main__':
    size, Y_select, bar = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    main(size, Y_select, bar)

# nohup python3 Signal.py 1 0 15 > models/1_0_15.log 2>&1 &
