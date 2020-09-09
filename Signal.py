import Data_Process, Model
import pandas as pd
import numpy as np
import h5py, sys
import datetime as dt
import matplotlib.pyplot as plt


def get_res(model, size, bar):
    d_res = Data_Process.mat_reader('data/Raw.mat')
    ticks_res = Data_Process.ticks_reader('data/w_data_ticks_15min.h5')
    trans_res = Data_Process.trans_reader('data/w_data_trans_15min.h5')

    # Find the dates of total dataset
    min_dates = 20160108
    max_dates = 20181231
    Data_Process.time_cut(d_res, ticks_res, trans_res, min_dates, max_dates)

    # Unzip
    _, dates, st_state, AdjustClse, clse, pclse, val, shr, TotalShares = d_res
    tickers, _, ask_order_volume_total, bid_order_volume_total, volume, close, pre_close, vwap = ticks_res
    _, _, amount_ask, amount_bid = trans_res

    amount_ask[np.isnan(amount_ask)] = 1
    amount_bid[np.isnan(amount_bid)] = 1

    # calculate the adjuested price, market value
    adjust_coef = AdjustClse / clse
    adj_close = close * adjust_coef[:, :, np.newaxis]
    adj_pre_close = pre_close * adjust_coef[:, :, np.newaxis]
    adj_vwap = vwap * adjust_coef[:, :, np.newaxis]

    # Get daily and intraday stock return
    diff_close_day = np.diff(adj_close, axis=0)
    close_ret = diff_close_day / adj_close[:-1]
    # close_ret_intra = adj_close[:, :, -1][:, :, np.newaxis] / adj_close - 1
    return_state = np.abs(clse / pclse - 1)
    daily_limit = (return_state > 0.095)

    # Get close daily and intraday market return (interday and intraday)
    stock_value = TotalShares * pclse
    market_value = np.nansum(stock_value, axis=1)
    stock_weights = stock_value / market_value[:, np.newaxis]
    market_ret = np.nansum(close_ret * stock_weights[1:, :, np.newaxis], axis=1)
    # market_ret_intra = np.nansum(close_ret_intra * stock_weights[:, :, np.newaxis], axis=1)

    features = np.array([ask_order_volume_total, bid_order_volume_total, volume,
                         adj_close, adj_pre_close, adj_vwap,
                         amount_ask, amount_bid])

    num_dates = dates.shape[0]
    num_tickers = tickers.shape[0]

    dataset = []
    input_shape = (size*16, 8)
    for date in range(num_dates-size+1):
        for t in range(num_tickers):
            st_series = st_state[date:(date+size), t]
            ele_data = features[:, date:(date + size), t].reshape((input_shape[1], input_shape[0]))
            if np.sum(st_series) == 0 and not np.isnan(return_state[date+size-1, t]) and \
                    not daily_limit[date+size-1, t] and not np.isnan(np.sum(ele_data)):
                ele_data = Data_Process.ele_normalize(ele_data, full=False)
            else:
                ele_data = np.full(input_shape, np.nan)
            dataset.append(ele_data)

    dataset = np.array(dataset)
    y_pred = model.predict(dataset).reshape((num_dates-size+1, num_tickers))
    print(y_pred.shape)
    y_true = close_ret[size-1:, :, bar] - market_ret[size-1:, bar][:, np.newaxis]
    print(y_true.shape)
    dates = dates[size-1:]
    return y_pred, y_true, dates, tickers


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
    res = []
    for index in indexs:
        y_pred_slice, y_true_slice = y_pred_cut[index], y_true[index]
        nan_count = np.sum(np.isnan(y_pred_slice))
        pred_rank = np.argsort(y_pred_slice)
        if nan_count > 0:
            pred_rank = pred_rank[:-nan_count]
        bins_average = divide_bins_average(y_true_slice, pred_rank)
        res.append(bins_average)
    res = np.array(res).T
    res = res + 1
    res = np.cumprod(res, axis=1)
    return res


def get_perform(y_pred, y_true, dates, train_start, valid_start, test_start, model_prefix):
    y_pred_cut, dates_cut = y_pred[:-1], dates[:-1]
    index_train = np.where((dates_cut < valid_start) & (dates_cut >= train_start))[0]
    index_valid = np.where((dates_cut < test_start) & (dates_cut >= valid_start))[0]
    index_test = np.where(dates_cut >= test_start)[0]
    train_res = get_perform_period(y_pred_cut, y_true, index_train)
    valid_res = get_perform_period(y_pred_cut, y_true, index_valid)
    test_res = get_perform_period(y_pred_cut, y_true, index_test)

    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(131)
    for i in range(len(train_res)):
        record = train_res[i]
        ax1.plot(record, label='group '+str(i))
    ax1.set_title('Train')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    for i in range(len(valid_res)):
        record = valid_res[i]
        ax2.plot(record, label='group '+str(i))
    ax2.set_title('Valid')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    for i in range(len(test_res)):
        record = test_res[i]
        ax3.plot(record, label='group '+str(i))
    ax3.set_title('Test')
    ax3.legend()

    fig.savefig('models/' + model_prefix + '_pnl.jpeg')


def main(size):
    input_shape = (size*16, 8)
    mode, float_init_lr, activation = 'x', 0.0001, 'relu'

    model_prefix = mode + '_' + str(size)
    model = Model.X_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                          num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                          kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1),
                          activation=activation)
    model.load('models/' + model_prefix + '_model.h5')
    y_pred, y_true, dates, tickers = get_res(model, size, 15)

    get_perform(y_pred, y_true, dates, 20160108, 20180528, 20180910, model_prefix)

    tickers_t = []
    for ticker in tickers:
        tickers_t.append(ticker.encode())

    with h5py.File('models/' + model_prefix + '_signal.h5', 'w') as h5f:
        h5f.create_dataset('signals', data=y_pred)
        h5f.create_dataset('return', data=y_true)
        h5f.create_dataset('dates', data=dates)
        h5f.create_dataset('tickers', data=tickers_t)


if __name__ == '__main__':
    size = int(sys.argv[1])
    main(size)

# nohup python3 Signal.py 6 > models/6.log 2>&1 &
