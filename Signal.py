import Data_Process, Model
import pandas as pd
import numpy as np
import h5py, sys
import datetime as dt
import matplotlib.pyplot as plt


def get_data(size, Y_select, bar, market):
    d_res = Data_Process.mat_reader('data/Raw.mat')
    ticks_res = Data_Process.ticks_reader('data/w_data_ticks_15min.h5')
    trans_res = Data_Process.trans_reader('data/w_data_trans_15min.h5')

    # Find the dates of total dataset
    min_dates = 20160108
    max_dates = 20191231
    Data_Process.time_cut(d_res, ticks_res, trans_res, min_dates, max_dates)

    print('d_res shape:', d_res[-1].shape)
    print('ticks_res shape:', ticks_res[-1].shape)
    print('trans_res shape:', trans_res[-1].shape)

    # Unzip
    _, dates, st_state, AdjustClse, clse, pclse, val, shr, TotalShares = d_res
    tickers, _, ask_order_volume_total, bid_order_volume_total, volume, close, pre_close, vwap = ticks_res
    _, _, amount_ask, amount_bid = trans_res

    amount_ask[np.isnan(amount_ask)] = 1
    amount_bid[np.isnan(amount_bid)] = 1

    # Judge whether there is mistake data at first!
    daily_ret = clse / pclse - 1
    daily_ret_mistake = np.where(np.abs(daily_ret) > 0.11)
    clse[daily_ret_mistake] = np.nan
    pclse[daily_ret_mistake] = np.nan
    AdjustClse[daily_ret_mistake] = np.nan

    close[np.where(np.abs(close / pre_close[:,:,0][:,:,np.newaxis] - 1) > 0.11)] = np.nan
    pre_close[np.where(np.abs(pre_close / pre_close[:,:,0][:,:,np.newaxis] - 1) > 0.11)] = np.nan
    vwap[np.where(np.abs(vwap / pre_close[:,:,0][:,:,np.newaxis] - 1) > 0.11)] = np.nan

    # Judge whether that bar is trade limit (If so, exclude from market return)
    bar_trade_limit = (np.abs(close / pre_close[:,:,0][:,:,np.newaxis] - 1) < 0.095)

    # calculate the adjuested price, market value
    adjust_coef = AdjustClse / clse
    adj_close = close * adjust_coef[:, :, np.newaxis]
    adj_pre_close = pre_close * adjust_coef[:, :, np.newaxis]
    adj_vwap = vwap * adjust_coef[:, :, np.newaxis]

    # Get daily and intraday stock return
    close_ret = (close[1:] / pre_close[1:,:,0][:,:,np.newaxis]) * \
                (close[:-1,:,-1][:,:,np.newaxis] / close[:-1]) - 1
    close_ret_intra = close[:, :, -1][:, :, np.newaxis] / close - 1

    # Get close daily and intraday market return (interday and intraday)
    # Get market return through market value weighted mean
    if market == 'm':
        stock_value = TotalShares * pclse
        market_value = np.nansum(stock_value, axis=1)
        stock_weights = stock_value / market_value[:, np.newaxis]
        market_ret = np.nansum(close_ret * bar_trade_limit[:-1] * stock_weights[1:, :, np.newaxis], axis=1)
        market_ret_intra = np.nansum(close_ret_intra * bar_trade_limit * stock_weights[:, :, np.newaxis], axis=1)

    # Get market return through simple mean
    else:
        market_ret = np.nanmean(close_ret * bar_trade_limit[:-1], axis=1)
        market_ret_intra = np.nanmean(close_ret_intra * bar_trade_limit, axis=1)

    # Get whether is able to trade
    trade_state = bar_trade_limit[:, :, bar]

    # Split dataset according to dates
    dataset = [np.array([st_state[:-1], trade_state[:-1]]),
               np.array([ask_order_volume_total[:-1], bid_order_volume_total[:-1], volume[:-1],
                         adj_close[:-1], adj_pre_close[:-1], adj_vwap[:-1],
                         amount_ask[:-1], amount_bid[:-1]]),
               np.array([close_ret, close_ret_intra[:-1]]),
               np.array([market_ret, market_ret_intra[:-1]])
               ]
    dates = dates[size:-1]
    return tickers, dates, dataset


def get_res(model, dataset, size, Y_select, bar):
    daily, features, stock_rets, market_rets = dataset[0], dataset[1], \
                                               dataset[2][Y_select,:,:,bar], dataset[3][Y_select,:,bar]
    rets = stock_rets - market_rets[:, np.newaxis]
    st_state, trade_state = daily[0], daily[1]
    input_shape = (16*size+bar+1, 8)
    nd, nt = st_state.shape
    signal_X = []
    for date in range(nd-size):
        for t in range(nt):
            st_series = st_state[date:(date+size+1), t]
            daily_part_ele_data = features[:, date:(date+size), t, :].reshape((8, -1))
            intra_part_ele_data = features[:, date+size, t, :(bar+1)].reshape((8, -1))
            ele_data = np.concatenate([daily_part_ele_data, intra_part_ele_data], axis=1)
            date_rets = rets[date+size, t]

            # Not ST, and not trading halt!
            if np.sum(st_series) == 0 and trade_state[date+size, t] and \
                    not np.isnan(np.sum(ele_data)) and not np.isnan(date_rets):
                ele_data = Data_Process.ele_normalize(ele_data, full=False)
                signal_X.append(ele_data)
            else:
                signal_X.append(np.full(input_shape, np.nan))

    y_true = rets[size:]
    signal_X = np.array(signal_X)
    y_pred = model.predict(signal_X).reshape(y_true.shape)
    return y_pred, y_true


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


def get_perform(y_pred, y_true, model_prefix):
    res, IC, w_IC = get_perform_period(y_pred, y_true)
    fig = plt.figure(figsize=(10,10))
    ax3 = fig.add_subplot(111)
    for i in range(len(res)):
        record = res[i]
        ax3.plot(record, label='group '+str(i))
    ax3.set_title('IC=%4f, All IC=%4f' % (IC, w_IC))
    ax3.legend()
    fig.savefig('models/' + model_prefix + '_pnl.jpeg')


def main(size, Y_select, bar, mode):
    input_shape = (size*16+bar+1, 8)
    tickers, dates, dataset = get_data(size, Y_select, bar, 'e')

    model_prefix = str(size) + '_' + str(Y_select) + '_' + str(bar) + '_' + mode
    if mode == 'cnn':
        model = Model.CNN_Pred(mode=mode, input_shape=input_shape, learning_rate=0.001,
                               num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                               kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1),
                               activation='relu')

    elif mode == 'tcn':
        model = Model.TCN_Model(mode=mode, input_shape=input_shape, learning_rate=0.001,
                                num_dense=16, activation='relu')

    elif mode == 'x':
        model = Model.X_Model(mode=mode, input_shape=input_shape, learning_rate=0.001,
                              num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                              kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1),
                              activation='relu')

    elif mode == 'y':
        model = Model.Y_Model(mode=mode, input_shape=input_shape, learning_rate=0.001,
                              num_vr_kernel=32, num_time_kernel=16, num_dense=16,
                              kernel_size=(2, 1), pool_size=(2, 1), strides=(2, 1),
                              activation='relu')

    else:
        model = Model.LSTM_Model(mode=mode, input_shape=input_shape, learning_rate=0.001,
                                 num_dense=16, activation='relu')

    model.load_model('models/' + model_prefix + '_model.h5')
    y_pred, y_true = get_res(model, dataset, size, Y_select, bar)

    get_perform(y_pred, y_true, model_prefix)

    tickers_t = []
    for ticker in tickers:
        tickers_t.append(ticker.encode())

    with h5py.File('models/' + model_prefix + '_signal.h5', 'w') as h5f:
        h5f.create_dataset('signals', data=y_pred)
        h5f.create_dataset('y_true', data=y_true)
        h5f.create_dataset('dates', data=dates)
        h5f.create_dataset('tickers', data=tickers_t)


if __name__ == '__main__':
    size, Y_select, bar, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    main(size, Y_select, bar, mode)

# nohup python3 Signal.py 4 0 15 x > models/4_0_15_x.log 2>&1 &
# nohup python3 Signal.py 3 1 2 y > models/3_1_2_y.log 2>&1 &
# nohup python3 Signal.py 5 0 15 cnn > models/5_0_15_cnn.log 2>&1 &
# nohup python3 Signal.py 5 1 2 cnn > models/5_1_2_cnn.log 2>&1 &
