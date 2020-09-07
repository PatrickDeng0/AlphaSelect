import Data_Process, Model
import pandas as pd
import numpy as np
import h5py, sys
import datetime as dt


def get_res(model, size):
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
            if np.sum(st_series) == 0:
                ele_data = features[:, date:(date + size), t].reshape((input_shape[1], input_shape[0]))
                if not np.isnan(np.sum(ele_data)):
                    ele_data = Data_Process.ele_normalize(ele_data, full=False)
                else:
                    ele_data = ele_data.T
            else:
                ele_data = np.full(input_shape, np.nan)
            dataset.append(ele_data)

    dataset = np.array(dataset)
    y_pred = model.predict(dataset).reshape((num_dates-size+1, num_tickers))
    dates = dates[size-1:]
    return y_pred, dates, tickers


def main(size):
    input_shape = (size*16, 8)
    mode, float_init_lr, activation = 'x', 0.0001, 'relu'

    model_prefix = mode + '_' + str(size)
    model = Model.X_Model(mode=mode, input_shape=input_shape, learning_rate=float_init_lr,
                          num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                          kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1),
                          activation=activation)
    model.load('models/' + model_prefix + '_model.h5')
    y_pred, dates, tickers = get_res(model, size)

    tickers_t = []
    for ticker in tickers:
        tickers_t.append(ticker.encode())

    with h5py.File('models/' + model_prefix + '_signal.h5', 'w') as h5f:
        h5f.create_dataset('signals', data=y_pred)
        h5f.create_dataset('dates', data=dates)
        h5f.create_dataset('tickers', data=tickers_t)


if __name__ == '__main__':
    size = int(sys.argv[1])
    main(size)

# nohup python3 Signal.py 6 > models/6.log 2>&1 &
