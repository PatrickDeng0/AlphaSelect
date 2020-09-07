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


def extract_2_X():
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

    # calculate the adjuested price, market value
    adjust_coef = AdjustClse / clse
    adj_close = close * adjust_coef[:, :, np.newaxis]
    adj_pre_close = pre_close * adjust_coef[:, :, np.newaxis]
    adj_vwap = vwap * adjust_coef[:, :, np.newaxis]

    # Get daily and intraday stock return
    diff_close_day = np.diff(adj_close, axis=0)
    close_ret = diff_close_day / adj_close[:-1]
    close_ret_intra = adj_close[:, :, -1][:, :, np.newaxis] / adj_close - 1

    # Get close daily and intraday market return (interday and intraday)
    stock_value = TotalShares * pclse
    market_value = np.nansum(stock_value, axis=1)
    stock_weights = stock_value / market_value[:, np.newaxis]
    market_ret = np.nansum(close_ret * stock_weights[1:, :, np.newaxis], axis=1)
    market_ret_intra = np.nansum(close_ret_intra * stock_weights[:, :, np.newaxis], axis=1)

    # Split dataset according to dates
    num_date = close_ret.shape[0]
    train_split = np.int(num_date * 0.8)
    test_split = np.int(num_date * 0.9)

    train_data = [[st_state[:train_split],
                   ask_order_volume_total[:train_split], bid_order_volume_total[:train_split], volume[:train_split],
                   adj_close[:train_split], adj_pre_close[:train_split], adj_vwap[:train_split],
                   amount_ask[:train_split], amount_bid[:train_split]],
                  np.array([close_ret[:train_split], close_ret_intra[:train_split]]),
                  np.array([market_ret[:train_split], market_ret_intra[:train_split]])]

    valid_data = [[st_state[train_split:test_split],
                   ask_order_volume_total[train_split:test_split], bid_order_volume_total[train_split:test_split],
                   volume[train_split:test_split], adj_close[train_split:test_split],
                   adj_pre_close[train_split:test_split], adj_vwap[train_split:test_split],
                   amount_ask[train_split:test_split], amount_bid[train_split:test_split]],
                  np.array([close_ret[train_split:test_split], close_ret_intra[train_split:test_split]]),
                  np.array([market_ret[train_split:test_split], market_ret_intra[train_split:test_split]])]

    # For only label_ret, the first axis is not the same as the features (-1)
    test_data = [[st_state[test_split:-1],
                  ask_order_volume_total[test_split:-1], bid_order_volume_total[test_split:-1], volume[test_split:-1],
                  adj_close[test_split:-1], adj_pre_close[test_split:-1], adj_vwap[test_split:-1],
                  amount_ask[test_split:-1], amount_bid[test_split:-1]],
                 np.array([close_ret[test_split:], close_ret_intra[test_split:-1]]),
                 np.array([market_ret[test_split:], market_ret_intra[test_split:-1]])]

    train_date = dates[:train_split]
    valid_date = dates[train_split:test_split]
    test_date = dates[test_split:-1]
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


# Somedays there are not valid returns
def check_mistake_rets(rets, m_rets):
    if np.isnan(np.sum(rets)) or np.max(np.abs(rets)) > 0.22 \
            or np.isnan(np.sum(m_rets)) or np.max(np.abs(m_rets)) > 0.22:
        return True
    return False


def X_cut(raw_data, size, train=False):
    features, stock_rets, market_rets = raw_data
    st_state = features[0]
    nd, nt = st_state.shape
    X, Y = [], []
    for t in range(nt):
        st_series = st_state[:, t]
        for date in range(nd-size+1):

            # Find that day: 20170105, at train data 242
            if train and 240 - size < date < 243:
                continue

            # Not ST, and not trading halt!
            if np.sum(st_series[date:(date+size)]) == 0:
                res, flag = [], False
                for i in range(1, len(features)):
                    item = features[i]
                    slice = item[date:(date + size), t].reshape(-1)

                    # if there is any data for ticks is np.nan, then next
                    # if ticks data is alright, then fill the nan to 0.0 for amount_bid or amount_ask data
                    if i < len(features) - 2:
                        if np.isnan(np.sum(slice)):
                            flag = True
                            break
                    else:
                        slice[np.isnan(slice)] = 1
                    res.append(slice)

                rets = stock_rets[:, date+size-1, t]
                m_rets = market_rets[:, date+size-1]

                # Judge whether record this data
                if not (flag or check_mistake_rets(rets, m_rets)):
                    X.append(np.vstack(res))
                    Y.append(rets - m_rets)
    return np.array(X), np.array(Y)


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


def dataset_normalize(dataset, select, start_bar, full=False):
    Y_select, bar = int(select), int(start_bar)
    if bar < 0 or bar > 15:
        print('Failed in Normalization because start_bar =', bar)
        return

    # Y_select == 0: predict interday return; == 1: predict intraday return
    # So now data normalize will reduce the time dimension by 16
    data_Y = dataset[1][:, Y_select, bar]
    if bar == 15:
        # predict the last bar
        data_X = dataset[0][:, :, (bar+1):]
    else:
        # predict the other bars
        last_bar = 15 - bar
        data_X = dataset[0][:, :, (bar+1):-last_bar]

    res_X = []
    for i in range(data_X.shape[0]):
        ele = data_X[i]
        res_X.append(ele_normalize(ele, full))
    return np.array(res_X), data_Y


def main(size):
    tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = extract_2_X()
    print('Extract Finish!')
    train_data = X_cut(train_data, size+1, train=True)
    valid_data = X_cut(valid_data, size+1)
    test_data = X_cut(test_data, size+1)

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(131)
    ax1 = sns.distplot(train_data[1][:, 0, :].reshape(-1))
    ax1.set_title('Train')

    ax2 = fig.add_subplot(132)
    ax2 = sns.distplot(valid_data[1][:, 0, :].reshape(-1))
    ax2.set_title('Valid')

    ax3 = fig.add_subplot(133)
    ax3 = sns.distplot(test_data[1][:, 0, :].reshape(-1))
    ax3.set_title('Test')
    fig.savefig('data/%d_interday_ret.jpeg' % size)

    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(131)
    ax1 = sns.distplot(train_data[1][:, 1, :].reshape(-1))
    ax1.set_title('Train')

    ax2 = fig.add_subplot(132)
    ax2 = sns.distplot(valid_data[1][:, 1, :].reshape(-1))
    ax2.set_title('Valid')

    ax3 = fig.add_subplot(133)
    ax3 = sns.distplot(test_data[1][:, 1, :].reshape(-1))
    ax3.set_title('Test')
    fig.savefig('data/%d_intraday_ret.jpeg' % size)
    return tickers, train_date, valid_date, test_date, train_data, valid_data, test_data


if __name__ == '__main__':
    main(2)
