import numpy as np
import pandas as pd
import random
import constants
import enums
import time
import pickle
import os
import datetime
import warnings
from datetime import timedelta
from scipy.optimize import minimize, differential_evolution, direct
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from functools import reduce
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


pd.options.mode.chained_assignment = None


seed = constants.RAMDOM_SEED



def reduction(arr, dicts, key):
    gap_lst = np.clip(arr, a_min=0, a_max=1)
    gap_lst = gap_lst / sum(gap_lst)
    deltas = np.cumsum(gap_lst)[:-1][::-1]
    deltas = scale(deltas, dicts, key)

    return deltas


def scale(arr, dicts, key):
    arr_lst = np.clip(arr, a_min=0, a_max=1)
    scale_lst = arr_lst * (dicts[f'{key}'][1] - dicts[f'{key}'][0]) + dicts[f'{key}'][0]

    return scale_lst


def transform(delta, window_size, X_df, y_df, tolerance):
    X_copy, y_copy = X_df.copy(), y_df.copy()

    df_lst = []
    for col in range(len(X_copy.columns) - 1):
        dfX_mini = X_copy[[f'{X_copy.columns[0]}', f'{X_copy.columns[col + 1]}']]

        dfX_mini.iloc[:, 0] = dfX_mini.iloc[:, 0] + timedelta(minutes=delta[col])
        dfX_mini = dfX_mini.rolling(f'{window_size:.0f}S', center=True, closed='both',
                                    on=f'{y_copy.columns[0]}').mean()

        df_mini = pd.merge_asof(y_copy, dfX_mini, on=f'{y_copy.columns[0]}', direction='nearest',
                                tolerance=pd.Timedelta(tolerance)).dropna(axis=0, how='any')
        df_mini.reset_index(inplace=True, drop=True)
        df_mini = df_mini.iloc[:, [0, -1]]
        df_lst.append(df_mini)

    df_lst.insert(0, y_copy)
    df = reduce(lambda left, right: pd.merge(left, right, on=[f'{y_copy.columns[0]}']), df_lst)
    cols = list(X_df.columns) + list(y_df.columns)[1:]
    final_df = df[cols]
    final_df = final_df.iloc[:, 0:]

    return final_df


def transform_data(delta, window_size, X_training, y_training, X_test, y_test, tolerance):
    data = transform(delta=delta, window_size=window_size, X_df=X_training, y_df=y_training, tolerance=tolerance)
    data_test = transform(delta=delta, window_size=window_size, X_df=X_test, y_df=y_test, tolerance=tolerance)

    return data, data_test


def cc_correlation(delta, window_size, X_df, y_df, tolerance):
    df = transform(delta, window_size, X_df, y_df, tolerance)
    corr_lst = []

    for feature in df.iloc[:, 1:-1].columns:
        corr_x = np.corrcoef(df[f'{feature}'], df.iloc[:, -1])[0, 1]
        corr_lst.append(abs(corr_x))

    return np.mean(corr_lst)


def mi_correlation(delta, window_size, X_df, y_df, tolerance):
    df = transform(delta, window_size, X_df, y_df, tolerance)
    corr_lst = []

    for feature in df.iloc[:, 1:-1].columns:
        X = np.array(df[f'{feature}']).reshape(-1, 1)
        corr_x = mutual_info_regression(X, df.iloc[:, -1],
                                        random_state=seed)
        corr_lst.append(abs(corr_x))

    return np.mean(corr_lst)


def classif_mi_correlation(delta, window_size, X_df, y_df, tolerance):
    df = transform(delta, window_size, X_df, y_df, tolerance=tolerance)
    corr_lst = []

    for feature in df.iloc[:, 1:-1].columns:
        X = np.array(df[f'{feature}']).reshape(-1, 1)
        y = df.iloc[:, -1]
        corr_x = mutual_info_classif(X, y, discrete_features=True, random_state=seed)
        corr_lst.append(abs(corr_x))

    return np.mean(corr_lst)


def opt_obj(x, dicts, X_df, y_df, method_name, tolerance):
    deltas = x[:-1]
    deltas = reduction(deltas, dicts, 'delta')

    window_sizes = x[-1]
    window_sizes = scale(window_sizes, dicts, 'window_size')

    if method_name == enums.CorrMethod.GCC.value:
        result = cc_correlation(deltas, window_sizes, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.TDMI.value:
        result = mi_correlation(deltas, window_sizes, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.CLASSIF_MI.value:
        result = classif_mi_correlation(deltas, window_sizes, X_df, y_df, tolerance)
    else:
        result = None

    return -result


def orderly_direct_opt(X_df, y_df, method_name, dicts, bounds, data_arrange, tolerance):
    starttime = time.time()

    res = differential_evolution(func=opt_obj, bounds=bounds, strategy='best1bin',
                                 args=(dicts, X_df, y_df, method_name, tolerance),
                                 disp=False, updating='deferred', workers=-1, maxiter=30)
    # res = direct(func=opt_obj, bounds=bounds, args=(dicts, X_df, y_df, method_name, tolerance), locally_biased=False, maxfun=100, maxiter=100, eps=1e-5)
    d_lst = res.x[:len(X_df.columns)]
    w_lst = res.x[-1]
    delta_opts = reduction(d_lst, dicts, 'delta')
    window_size_opts = scale(w_lst, dicts, 'window_size')

    if method_name == enums.CorrMethod.GCC.value:
        values = cc_correlation(delta_opts, window_size_opts, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.TDMI.value:
        values = mi_correlation(delta_opts, window_size_opts, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.CLASSIF_MI.value:
        values = classif_mi_correlation(delta_opts, window_size_opts, X_df, y_df, tolerance)
    else:
        raise RuntimeError(f'cannot find method_name: {method_name}')

    nfev = res['nfev']
    message = res['message']

    endtime = time.time()
    result_dict = {'avg_delta': delta_opts, 'avg_window_size': window_size_opts, 'avg_value': values,
                   'avg_nfev': nfev, 'run_time': endtime - starttime, 'decimal_delta': d_lst,
                   'decimal_window_size': w_lst, 'message': message}

    times = 1
    prefix = f'orderly_{data_arrange}_training'
    file = f'{prefix}_col{len(X_df.columns)}_{method_name}_test{times}.pkl'
    while os.path.exists(file):
        times += 1
        file = file.replace(f"_test{times - 1}.pkl", f"_test{times}.pkl")
    with open(file, mode="wb") as f:
        pickle.dump(result_dict, f, True)


    return delta_opts, window_size_opts, values, nfev, d_lst, w_lst


def bootstrap_orderly_direct_opt(X_df, y_df, method_name, dicts, bounds, tolerance):
    # res = differential_evolution(func=opt_obj, bounds=bounds, strategy='best1bin',
    #                              args=(dicts, X_df, y_df, method_name, tolerance),
    #                              disp=False, updating='deferred', workers=-1, maxiter=30)
    res = direct(func=opt_obj, bounds=bounds, args=(dicts, X_df, y_df, method_name, tolerance), locally_biased=False, maxfun=100, maxiter=100, eps=1e-5)
    d_lst = res.x[:len(X_df.columns)]
    w_lst = res.x[-1]
    delta_opts = reduction(d_lst, dicts, 'delta')
    window_size_opts = scale(w_lst, dicts, 'window_size')
    if method_name == enums.CorrMethod.GCC.value:
        values = cc_correlation(delta_opts, window_size_opts, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.TDMI.value:
        values = mi_correlation(delta_opts, window_size_opts, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.CLASSIF_MI.value:
        values = classif_mi_correlation(delta_opts, window_size_opts, X_df, y_df, tolerance)
    else:
        raise RuntimeError(f'cannot find method_name: {method_name}')
    nfev = res['nfev']

    return delta_opts, window_size_opts, values, nfev, d_lst, w_lst


def bootstrap_transform(drop_rate, X_df):
    real_X_copy = X_df.copy()
    time_lst = list(real_X_copy.iloc[:, 0])
    retain_nums = round(len(time_lst) * (1 - drop_rate))
    drop_nums = round(len(time_lst) * drop_rate)
    node = random.randint(0, retain_nums)
    X_cut_df = real_X_copy.iloc[node:node + drop_nums, :]

    return X_cut_df


def bootstrap_df_concat(drop_rate, X_df, y_df):
    current_year = X_df.iloc[:, 0].dt.year.iloc[0]
    X_df_lst = [bootstrap_transform(drop_rate, X_df) for i in range(int(1 // drop_rate))]
    y_df_lst = []
    for i in range(len(X_df_lst)):
        real_y_copy = y_df.copy()
        X_df_lst[i].iloc[:, 0] = X_df_lst[i].iloc[:, 0].apply(lambda x: x.replace(year=current_year + i * 10))
        real_y_copy.iloc[:, 0] = real_y_copy.iloc[:, 0].apply(lambda x: x.replace(year=current_year + i * 10))
        y_df_lst.append(real_y_copy)

    X_concat_df, y_concat_df = pd.concat(X_df_lst, axis=0), pd.concat(y_df_lst, axis=0)

    return X_concat_df, y_concat_df


def bootstrap_corr(drop_rate, nums, X_df, y_df, method_name, dicts, bounds, data_arrange, tolerance):
    starttime = time.time()
    delta_lst = []
    window_size_lst = []
    nfev_lst = []
    bootstrap_lst = []
    decimal_deltas_lst = []
    decimal_window_size_lst = []
    bootstrap_decimal_lst = []

    for i in range(nums):
        X_concat_df, y_concat_df = bootstrap_df_concat(drop_rate=drop_rate, X_df=X_df, y_df=y_df)
        X_concat_df = X_concat_df.sort_values([X_concat_df.columns[0]])
        y_concat_df = y_concat_df.sort_values([y_concat_df.columns[0]])
        deltas, window_sizes, value, nfev, decimal_delta, decimal_window_size = bootstrap_orderly_direct_opt(
            X_df=X_concat_df,
            y_df=y_concat_df,
            method_name=method_name,
            dicts=dicts,
            bounds=bounds, tolerance=tolerance)

        r_dict = {'delta': deltas, 'window_size': window_sizes, 'value': value, 'nfev': nfev}
        bootstrap_lst.append(r_dict)
        decimal_dict = {'decimal_delta': decimal_delta, 'decimal_window_size': decimal_window_size}
        bootstrap_decimal_lst.append(decimal_dict)

        delta_lst.append(deltas)
        window_size_lst.append(window_sizes)
        nfev_lst.append(nfev)
        decimal_deltas_lst.append(decimal_delta)
        decimal_window_size_lst.append(decimal_window_size)
        print(f'The {i} th iteration has been completed')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


    delta_mean = np.array(delta_lst).mean(axis=0)
    window_size_mean = ((np.array(window_size_lst))).mean(axis=0)
    if method_name == enums.CorrMethod.GCC.value:
        values = cc_correlation(delta_mean, window_size_mean, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.TDMI.value:
        values = mi_correlation(delta_mean, window_size_mean, X_df, y_df, tolerance)
    elif method_name == enums.CorrMethod.CLASSIF_MI.value:
        values = classif_mi_correlation(delta_mean, window_size_mean, X_df, y_df, tolerance)
    else:
        raise RuntimeError(f'cannot find method_name: {method_name}')

    nfev = np.array(nfev_lst).mean(axis=0)
    decimal_delta_mean = np.array(decimal_deltas_lst).mean(axis=0)
    decimal_window_size_mean = np.array(decimal_window_size_lst).mean(axis=0)

    endtime = time.time()

    result_dict = {'avg_delta': delta_mean, 'avg_window_size': window_size_mean, 'avg_value': values, 'avg_nfev': nfev,
                   'run_time': endtime - starttime, 'decimal_delta': decimal_delta_mean,
                   'decimal_window_size': decimal_window_size_mean,
                   'bootstrap_process': bootstrap_lst, 'bootstrap_decimal_process': bootstrap_decimal_lst}

    times = 1
    file = f'bootstrap_{data_arrange}_training_col{len(X_df.columns)}_{method_name}_nums{nums}_test{times}.pkl'
    if os.path.exists(file) == False:
        with open(file=file, mode="wb") as f:
            pickle.dump(result_dict, f, True)
    else:
        while os.path.exists(file):
            times += 1
            new_file = f'bootstrap_{data_arrange}_training_col{len(X_df.columns)}_{method_name}_nums{nums}_test{times}.pkl'
            if os.path.exists(new_file) == False:
                with open(file=new_file, mode="wb") as f:
                    pickle.dump(result_dict, f, True)
            else:
                continue
            break

    return delta_mean, window_size_mean, values, nfev, decimal_delta_mean, decimal_window_size_mean


def train_test_splits(X_df, y_df, test_rate, tolerance):
    X_copy, y_copy = X_df.copy(), y_df.copy()

    y_length = round(len(y_copy) * (1 - test_rate))

    y_training = y_copy.iloc[:y_length, :]
    y_test = y_copy.iloc[y_length:, :]

    training = pd.merge_asof(y_training, X_copy, on=f'{y_copy.columns[0]}', direction='nearest',
                             tolerance=pd.Timedelta(tolerance)).dropna(axis=0, how='any')
    X_training = training[X_copy.columns]
    y_training = training[y_copy.columns]

    test = pd.merge_asof(y_test, X_copy, on=f'{y_copy.columns[0]}', direction='nearest',
                         tolerance=pd.Timedelta(tolerance)).dropna(axis=0, how='any')
    X_test = test[X_copy.columns]
    y_test = test[y_copy.columns]

    return X_training, X_test, y_training, y_test