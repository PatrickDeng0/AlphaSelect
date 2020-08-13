import numpy as np
import util
import h5py
import datetime as dt


def mat_reader(filename):
    with h5py.File(filename) as mat:
        mat_data = mat['StockMat']
        mat_tkrs = mat_data['tkrs'][0]
        tickers = []
        for i in range(mat_tkrs.shape[0]):
            st = mat_tkrs[i]
            obj = mat_data[st]
            string = "".join(chr(i) for i in obj[:])
            tickers.append(string)
        tickers = np.array(tickers)

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
    min_dates = np.max([d_res[1].min(), ticks_res[1].min(), trans_res[1].min()])
    max_dates = np.min([d_res[1].max(), ticks_res[1].max(), trans_res[1].max()])
    time_cut(d_res, ticks_res, trans_res, min_dates, max_dates)

    # Unzip
    tickers, dates, st_state, AdjustClse, clse, pclse, val, shr, TotalShares = d_res
    _, _, ask_order_volume_total, bid_order_volume_total, volume, close, pre_close, vwap = ticks_res
    _, _, amount_ask, amount_bid = trans_res

    # calculate the adjuested price and its return
    adjust_coef = AdjustClse / clse
    adj_close = close * adjust_coef[:, :, np.newaxis]
    adj_pre_close = pre_close * adjust_coef[:, :, np.newaxis]
    adj_vwap = vwap * adjust_coef[:, :, np.newaxis]
    diff_vwap = np.diff(adj_vwap, axis=0)
    vwap_ret = diff_vwap / adj_vwap[:-1]

    # Get market return
    stock_value = TotalShares[:, :, np.newaxis] * vwap
    market_value = np.nansum(stock_value, axis=1)
    diff_market = np.diff(market_value, axis=0)
    market_ret = diff_market / market_value[:-1]

    # Label return
    label_ret = vwap_ret - market_ret[:, np.newaxis, :]

    # Split dataset according to dates
    num_date = label_ret.shape[0]
    train_split = np.int(num_date * 0.8)
    test_split = np.int(num_date * 0.9)

    train_data = [[st_state[:train_split],
                   ask_order_volume_total[:train_split], bid_order_volume_total[:train_split], volume[:train_split],
                   adj_close[:train_split], adj_pre_close[:train_split], adj_vwap[:train_split],
                   amount_ask[:train_split], amount_bid[:train_split]],
                  label_ret[:train_split]]

    valid_data = [[st_state[train_split:test_split],
                   ask_order_volume_total[train_split:test_split], bid_order_volume_total[train_split:test_split],
                   volume[train_split:test_split], adj_close[train_split:test_split],
                   adj_pre_close[train_split:test_split], adj_vwap[train_split:test_split],
                   amount_ask[train_split:test_split], amount_bid[train_split:test_split]],
                  label_ret[train_split:test_split]]

    # For only label_ret, the first axis is not the same as the features (-1)
    test_data = [[st_state[test_split:-1],
                  ask_order_volume_total[test_split:-1], bid_order_volume_total[test_split:-1], volume[test_split:-1],
                  adj_close[test_split:-1], adj_pre_close[test_split:-1], adj_vwap[test_split:-1],
                  amount_ask[test_split:-1], amount_bid[test_split:-1]],
                 label_ret[test_split:]]

    train_date = dates[:train_split]
    valid_date = dates[train_split:test_split]
    test_date = dates[test_split:-1]
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


# ele data structure:
# ask_order_volume_total, bid_order_volume_total, volume (Volume related)
# adj_close, adj_pre_close, adj_vwap (price related)
# amount_ask, amount_bid (Amount related)

def normalize(ele):
    stand_price, stand_val, stand_volume = ele[5].mean(), ele[7].mean(), ele[2].mean()
    X = np.vstack([ele[0]/stand_volume, ele[1]/stand_volume, ele[2]/stand_volume,
                   ele[3]/stand_price, ele[4]/stand_price, ele[5]/stand_price,
                   ele[6]/stand_val, ele[7]/stand_val])
    return X.T


def X_cut(raw_data, size):
    features, label_ret = raw_data
    st_state = features[0]
    nd, nt = st_state.shape
    X, Y = [], []
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
                    if np.sum(np.isnan(slice)) > 0:
                        flag = True
                        break
                    res.append(slice)

                # if there is any return data is np.nan, then next
                vwap_ret = label_ret[date+size-1, t, -1]
                if not flag and not np.isnan(vwap_ret):
                    X.append(normalize(res))
                    Y.append(vwap_ret)
    return np.array(X), np.array(Y)


def binarize_data(dataset):
    binary_y = (dataset[1] > 0) + 0
    return dataset[0], binary_y


def main(size):
    tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = extract_2_X()
    print('Extract Finish!')
    train_data = X_cut(train_data, size)
    valid_data = X_cut(valid_data, size)
    test_data = X_cut(test_data, size)
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


if __name__ == '__main__':
    pass
