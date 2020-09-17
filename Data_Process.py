import numpy as np
import h5py, sys
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


def mat_reader(filename):
    with h5py.File(filename) as mat:
        mat_data = mat['StockMat']
        tickers = []
        dates = mat_data['dtes'].value[0].astype(np.int32)
        st_state = mat_data['STState'].value
        clse = mat_data['clse'].value
        pclse = mat_data['pclse'].value
        AdjustClse = mat_data['AdjustedClse'].value
        val = mat_data['val'].value
        shr = mat_data['shr'].value
        TotalShares = mat_data['TotalShares'].value
    return [tickers, dates, st_state, AdjustClse, clse, pclse, val, shr, TotalShares]


def ticks_reader(filename):
    with h5py.File(filename) as ticks:
        ask_order_volume_total = ticks['ask_order_volume_total'].value
        bid_order_volume_total = ticks['bid_order_volume_total'].value
        close = ticks['close'].value
        dates = ticks['dates'].value
        pre_close = ticks['pre_close'].value
        tickers = ticks['tickers'].value.astype(str)
        volume = ticks['volume'].value
        vwap = ticks['vwap'].value
    return [tickers, dates, ask_order_volume_total, bid_order_volume_total, volume, close, pre_close, vwap]


def trans_reader(filename):
    with h5py.File(filename) as trans:
        amount_ask = trans['amount_ask'].value
        amount_bid = trans['amount_bid'].value
        dates = trans['dates'].value
        tickers = trans['tickers'].value.astype(str)
    return [tickers, dates, amount_ask, amount_bid]


def time_cut(d_res, ticks_res, trans_res, min_dates, max_dates):
    def index_cut(res, cut_array, daily=False):
        for i in range(1, len(res)):
            if i == 1:
                res[i] = res[i][cut_array]
            else:
                if daily:
                    res[i] = res[i].swapaxes(0,1)
                else:
                    res[i] = res[i].swapaxes(1,2)
                res[i] = res[i][cut_array, :]

    res_d_dates = np.where((d_res[1] >= min_dates) & (d_res[1] <= max_dates))[0]
    res_ticks_dates = np.where((ticks_res[1] >= min_dates) & (ticks_res[1] <= max_dates))[0]
    res_trans_dates = np.where((trans_res[1] >= min_dates) & (trans_res[1] <= max_dates))[0]
    index_cut(d_res, res_d_dates, daily=True)
    index_cut(ticks_res, res_ticks_dates, daily=False)
    index_cut(trans_res, res_trans_dates, daily=False)


def extract_2_X(size, Y_select, bar, market):
    d_res = mat_reader('data/Raw.mat')
    ticks_res = ticks_reader('data/w_data_ticks_15min.h5')
    trans_res = trans_reader('data/w_data_trans_15min.h5')

    # Find the dates of total dataset
    min_dates = 20160108
    max_dates = 20181231
    time_cut(d_res, ticks_res, trans_res, min_dates, max_dates)

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
    dates = dates[:-1]
    train_split = np.sum(dates <= 20180000)
    test_split = np.sum(dates <= 20180699)
    return tickers, dates, dataset, train_split, test_split


def X_cut(dataset, start, end, size, Y_select, bar):
    daily, features, stock_rets, market_rets = dataset[0][:,start:end], dataset[1][:,start:end], \
                                               dataset[2][Y_select,start:end,:,bar], dataset[3][Y_select,start:end,bar]
    rets = stock_rets - market_rets[:, np.newaxis]
    st_state, trade_state = daily[0], daily[1]
    input_shape = (16*size+bar+1, 8)
    nd, nt = st_state.shape
    X, Y = [], []
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
                ele_data = ele_normalize(ele_data, full=False)
                X.append(ele_data)
                signal_X.append(ele_data)
                Y.append(date_rets)
            else:
                signal_X.append(np.full(input_shape, np.nan))

    signal_X = np.array(signal_X)
    signal_Y = rets[size:]
    return (np.array(X), np.array(Y)), (signal_X, signal_Y)


# ele data structure:
# ask_order_volume_total, bid_order_volume_total, volume (Volume related)
# adj_close, adj_pre_close, adj_vwap (price related)
# amount_ask, amount_bid (Amount related)

def ele_normalize(ele, full):
    volume_total = ele[0] + ele[1]
    amount_total = ele[6] + ele[7]
    stand_price, stand_val, stand_volume = ele[3].mean(), ele[7].mean(), ele[2].mean()
    if full:
        X = np.vstack([ele[0]/volume_total - 0.5, ele[1]/volume_total - 0.5, ele[2]/stand_volume - 1,
                       ele[3]/stand_price - 1, ele[4]/stand_price - 1, ele[5]/stand_price - 1,
                       ele[6]/amount_total - 0.5, ele[7]/amount_total - 0.5,
                       ele[0]/stand_volume, ele[1]/stand_volume,
                       ele[6]/stand_val, ele[7]/stand_val])
    else:
        X = np.vstack([ele[0]/volume_total - 0.5, ele[1]/volume_total - 0.5, ele[2]/stand_volume - 1,
                       ele[3]/stand_price - 1, ele[4]/stand_price - 1, ele[5]/stand_price - 1,
                       ele[6]/amount_total - 0.5, ele[7]/amount_total - 0.5])
    return X.T


def main(size, Y_select, bar, market):
    tickers, dates, dataset, train_split, test_split = extract_2_X(size, Y_select, bar, market)
    end_dateset = dataset[0].shape[1]
    print('Extract Finish!')
    train_data = X_cut(dataset, 0, train_split, size, Y_select, bar)
    valid_data = X_cut(dataset, train_split, test_split, size, Y_select, bar)
    test_data = X_cut(dataset, test_split, end_dateset, size, Y_select, bar)

    train_date = dates[:train_split]
    valid_date = dates[train_split:test_split]
    test_date = dates[test_split:]

    # fig = plt.figure(figsize=(18,6))
    # ax1 = fig.add_subplot(131)
    # ax1 = sns.distplot(train_data[1][:, 0, :].reshape(-1))
    # ax1.set_title('Train')
    #
    # ax2 = fig.add_subplot(132)
    # ax2 = sns.distplot(valid_data[1][:, 0, :].reshape(-1))
    # ax2.set_title('Valid')
    #
    # ax3 = fig.add_subplot(133)
    # ax3 = sns.distplot(test_data[1][:, 0, :].reshape(-1))
    # ax3.set_title('Test')
    # fig.savefig('data/%d_interday_ret.jpeg' % size)
    #
    # fig = plt.figure(figsize=(18,6))
    # ax1 = fig.add_subplot(131)
    # ax1 = sns.distplot(train_data[1][:, 1, :].reshape(-1))
    # ax1.set_title('Train')
    #
    # ax2 = fig.add_subplot(132)
    # ax2 = sns.distplot(valid_data[1][:, 1, :].reshape(-1))
    # ax2.set_title('Valid')
    #
    # ax3 = fig.add_subplot(133)
    # ax3 = sns.distplot(test_data[1][:, 1, :].reshape(-1))
    # ax3.set_title('Test')
    # fig.savefig('data/%d_intraday_ret.jpeg' % size)
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


if __name__ == '__main__':
    main(2, 0, 15, 'e')
