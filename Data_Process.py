import numpy as np
import h5py
import datetime as dt


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


def extract_2_X():
    d_res = mat_reader('data/Raw.mat')
    ticks_res = ticks_reader('data/w_data_ticks_15min.h5')
    trans_res = trans_reader('data/w_data_trans_15min.h5')

    # Find the dates of total dataset
    min_dates = 20160101
    max_dates = np.min([d_res[1].max(), ticks_res[1].max(), trans_res[1].max()])
    time_cut(d_res, ticks_res, trans_res, min_dates, max_dates)

    print('d_res shape:', d_res[-1].shape)
    print('ticks_res shape:', ticks_res[-1].shape)
    print('trans_res shape:', trans_res[-1].shape)

    # Unzip
    _, dates, st_state, AdjustClse, clse, pclse, val, shr, TotalShares = d_res
    tickers, _, ask_order_volume_total, bid_order_volume_total, volume, close, pre_close, vwap = ticks_res
    _, _, amount_ask, amount_bid = trans_res

    # calculate the adjuested price, market value
    adjust_coef = AdjustClse / clse
    adj_close = close * adjust_coef[:, :, np.newaxis]
    adj_pre_close = pre_close * adjust_coef[:, :, np.newaxis]
    adj_vwap = vwap * adjust_coef[:, :, np.newaxis]
    stock_value = TotalShares[:, :, np.newaxis] * vwap
    market_value = np.nansum(stock_value, axis=1)

    # Get daily stock return and market return
    diff_vwap_day = np.diff(adj_vwap, axis=0)
    vwap_ret = diff_vwap_day / adj_vwap[:-1]
    diff_market = np.diff(market_value, axis=0)
    market_ret = diff_market / market_value[:-1]
    label_ret = vwap_ret - market_ret[:, np.newaxis, :]

    # Get intraday stock return and market return
    vwap_ret_intra = adj_vwap[:, :, -1][:, :, np.newaxis] / adj_vwap - 1
    market_ret_intra = market_value[:, -1][:, np.newaxis] / market_value - 1
    label_ret_intra = vwap_ret_intra - market_ret_intra[:, np.newaxis, :]

    # Split dataset according to dates
    num_date = label_ret.shape[0]
    train_split = np.int(num_date * 0.8)
    test_split = np.int(num_date * 0.9)

    train_data = [[st_state[:train_split],
                   ask_order_volume_total[:train_split], bid_order_volume_total[:train_split], volume[:train_split],
                   adj_close[:train_split], adj_pre_close[:train_split], adj_vwap[:train_split],
                   amount_ask[:train_split], amount_bid[:train_split]],
                  [label_ret[:train_split], label_ret_intra[:train_split]]]

    valid_data = [[st_state[train_split:test_split],
                   ask_order_volume_total[train_split:test_split], bid_order_volume_total[train_split:test_split],
                   volume[train_split:test_split], adj_close[train_split:test_split],
                   adj_pre_close[train_split:test_split], adj_vwap[train_split:test_split],
                   amount_ask[train_split:test_split], amount_bid[train_split:test_split]],
                  [label_ret[train_split:test_split], label_ret_intra[train_split:test_split]]]

    # For only label_ret, the first axis is not the same as the features (-1)
    test_data = [[st_state[test_split:-1],
                  ask_order_volume_total[test_split:-1], bid_order_volume_total[test_split:-1], volume[test_split:-1],
                  adj_close[test_split:-1], adj_pre_close[test_split:-1], adj_vwap[test_split:-1],
                  amount_ask[test_split:-1], amount_bid[test_split:-1]],
                 [label_ret[test_split:], label_ret_intra[test_split:-1]]]

    train_date = dates[:train_split]
    valid_date = dates[train_split:test_split]
    test_date = dates[test_split:-1]
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


def X_cut(raw_data, size):
    features, label_rets = raw_data
    st_state = features[0]
    nd, nt = st_state.shape
    X, Y, Y_intra = [], [], []
    for t in range(nt):
        if t % 300 == 0:
            print('Now stock', t, dt.datetime.now())
        st_series = st_state[:, t]
        for date in range(nd-size+1):
            # Not ST, and not trading halt!
            if np.sum(st_series[date:(date+size)]) == 0:
                res, flag = [], False
                for item in features[1:]:
                    slice = item[date:(date + size), t].reshape(-1)

                    # if there is any data is np.nan, then next
                    if np.isnan(np.sum(slice)):
                        flag = True
                        break
                    res.append(slice)

                # if there is any return data is np.nan, then next
                vwap_ret = label_rets[0][date+size-1, t]
                vwap_ret_intra = label_rets[1][date+size-1, t]

                # Not np.nan
                # Y: the return of interday return for every bar
                # Y_intra: the return of intraday return for every bar (compared to that day's last bar)
                if not (flag or np.isnan(np.sum(vwap_ret)) or np.isnan(np.sum(vwap_ret_intra))):
                    X.append(np.vstack(res))
                    Y.append(vwap_ret)
                    Y_intra.append(vwap_ret_intra)
    return np.array(X), np.array(Y), np.array(Y_intra)


# ele data structure:
# ask_order_volume_total, bid_order_volume_total, volume (Volume related)
# adj_close, adj_pre_close, adj_vwap (price related)
# amount_ask, amount_bid (Amount related)

def ele_normalize(ele, full):
    volume_total = ele[0] + ele[1]
    amount_total = ele[6] + ele[7]
    stand_price, stand_val, stand_volume = ele[5].mean(), ele[7].mean(), ele[2].mean()
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


def dataset_normalize(dataset, intra, start_bar, full=False):
    intra, bar = int(intra), int(start_bar)
    if bar < 0 or bar > 15:
        print('Failed in Normalization because start_bar =', bar)
        return

    if intra == 0:
        # predict interday return
        Y_select = 1
    else:
        # predict intraday return
        Y_select = 2

    if bar == 15:
        # predict the last bar
        data_X, data_Y = dataset[0], dataset[Y_select][:, bar]
    else:
        # predict the other bars
        last_bar = 15 - bar
        data_X, data_Y = dataset[0][:,:,:-last_bar], dataset[Y_select][:, bar]

    res_X = []
    for ele in data_X:
        res_X.append(ele_normalize(ele, full))
    return np.array(res_X), data_Y


def main(size):
    tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = extract_2_X()
    print('Extract Finish!')
    train_data = X_cut(train_data, size)
    valid_data = X_cut(valid_data, size)
    test_data = X_cut(test_data, size)
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


if __name__ == '__main__':
    pass
