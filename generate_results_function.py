import heapq
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import constants
import model_utils
import opt_utils
import enums
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

seed = constants.RAMDOM_SEED


def tsmb_method(val_df_lst, test_df_lst, n, y_test, task_name):
    y_test_copy = y_test.copy()
    final_indicator_lst = []

    if task_name == enums.TaskType.CLASSIFICATION.value:
        val_column_prefix = 'val_auc'
        test_column_prefix = 'test_y_pred_proba1_'
    elif task_name == enums.TaskType.REGRESSION.value:
        val_column_prefix = 'val_r2'
        test_column_prefix = 'test_y_pred_'
    else:
        raise ValueError("Invalid task_name value. Expected 'classification' or 'regression'.")

    for run_id in range(constants.BOOTSTRAP_ITERATIONS):
        val_indicator_lst = [float(val_df_lst[run_id][i].mode()) for i in val_df_lst[run_id].columns if i.startswith(val_column_prefix)]
        data = heapq.nlargest(n, enumerate(val_indicator_lst), key=lambda x: x[1])
        index, values = zip(*data)

        pred_lst = [test_df_lst[run_id][f'{test_column_prefix}{j}'] for j in index]
        pred_df = pd.DataFrame(pred_lst).T
        pred_mean_df = pd.DataFrame({'time': test_df_lst[run_id].iloc[:, 0], 'y_pred': pred_df.mean(axis=1)})
        pred_mean_df = pred_mean_df.sort_values(['time']).dropna()
        y_test_copy.columns = ['time', 'y_true']
        final_df = pd.merge(y_test_copy, pred_mean_df, on='time', how='left').dropna()

        if task_name == enums.TaskType.CLASSIFICATION.value:
            final_auc = metrics.roc_auc_score(y_true=final_df.iloc[:, 1], y_score=final_df.iloc[:, 2])
            final_indicator_lst.append(final_auc)
        elif task_name == enums.TaskType.REGRESSION.value:
            final_r2 = r2_score(y_true=final_df.iloc[:, 1], y_pred=final_df.iloc[:, 2])
            final_indicator_lst.append(final_r2)

    return final_indicator_lst


def tdb_method(val_df_lst, b_lst, n, X_training, y_training, X_test, y_test, task_name, tolerance):
    final_indicator_lst = []

    for run_id in range(constants.BOOTSTRAP_ITERATIONS):
        if task_name == enums.TaskType.CLASSIFICATION.value:
            val_auc_lst = [float(val_df_lst[run_id][i].mode()) for i in val_df_lst[run_id].columns if
                           i[0:7] == 'val_auc']
            data = heapq.nlargest(n, enumerate(val_auc_lst), key=lambda x: x[1])
            index, values = zip(*data)

            d_lst = [b_lst[run_id]['bootstrap_process'][j]['delta'] for j in index]
            w_lst = [b_lst[run_id]['bootstrap_process'][j]['window_size'] for j in index]
            final_d = np.array(d_lst).mean(axis=0)
            final_w = np.array(w_lst).mean(axis=0)
            final_data, final_data_test = opt_utils.transform_data(final_d, final_w, X_training, y_training, X_test, y_test, tolerance)

            results = model_utils.gdbt_classifier_model(final_data, final_data_test)
            final_indicator_lst.append(results[0])

        elif task_name == enums.TaskType.REGRESSION.value:
            val_r2_lst = [float(val_df_lst[run_id][i].mode()) for i in val_df_lst[run_id].columns if i[0:6] == 'val_r2']
            data = heapq.nlargest(n, enumerate(val_r2_lst), key=lambda x: x[1])
            index, values = zip(*data)

            d_lst = [b_lst[run_id]['bootstrap_process'][j]['delta'] for j in index]
            w_lst = [b_lst[run_id]['bootstrap_process'][j]['window_size'] for j in index]
            final_d = np.array(d_lst).mean(axis=0)
            final_w = np.array(w_lst).mean(axis=0)
            final_data, final_data_test = opt_utils.transform_data(final_d, final_w, X_training, y_training, X_test, y_test, tolerance)

            results = model_utils.gdbt_regressor_model(final_data, final_data_test)
            final_indicator_lst.append(results[0])

        else:
            raise RuntimeError(f'cannot find task_name: {task_name}')

    return final_indicator_lst


def cofidence_func(test_df_lst, y_test, task_name):
    rate_95_lst = []
    rate_90_lst = []
    rate_80_lst = []

    for i in range(constants.BOOTSTRAP_ITERATIONS):
        cc = test_df_lst[i].copy()
        y_pred_lst = []
        for j in cc.columns:
            if task_name == enums.TaskType.REGRESSION.value:
                if j[0:11] == 'test_y_pred':
                    y_pred_lst.append(j)
            elif task_name == enums.TaskType.CLASSIFICATION.value:
                if j[0:18] == 'test_y_pred_proba1':
                    y_pred_lst.append(j)

        y_pred_lst.insert(0, 'time')
        cc_mini = cc[y_pred_lst]
        cc_mini['95_lower'] = cc_mini.quantile(0.025, axis=1)
        cc_mini['95_higher'] = cc_mini.quantile(0.975, axis=1)
        cc_mini['90_lower'] = cc_mini.quantile(0.05, axis=1)
        cc_mini['90_higher'] = cc_mini.quantile(0.95, axis=1)
        cc_mini['80_lower'] = cc_mini.quantile(0.1, axis=1)
        cc_mini['80_higher'] = cc_mini.quantile(0.9, axis=1)

        label_df_copy = y_test.copy()
        label_df_copy.columns = ['time', 'test_y_true']
        final_mini_df = pd.merge(label_df_copy, cc_mini, on='time', how='left')
        final_mini_df.dropna(inplace=True)
        if task_name == enums.TaskType.REGRESSION.value:
            rate_95 = sum((final_mini_df['test_y_true'] >= final_mini_df['95_lower']) & (
                    final_mini_df['test_y_true'] <= final_mini_df['95_higher'])) / len(final_mini_df)
            rate_90 = sum((final_mini_df['test_y_true'] >= final_mini_df['90_lower']) & (
                    final_mini_df['test_y_true'] <= final_mini_df['90_higher'])) / len(final_mini_df)
            rate_80 = sum((final_mini_df['test_y_true'] >= final_mini_df['80_lower']) & (
                    final_mini_df['test_y_true'] <= final_mini_df['80_higher'])) / len(final_mini_df)
        elif task_name == enums.TaskType.CLASSIFICATION.value:
            round_95_lower = final_mini_df['95_lower'].round().astype(int)
            round_95_higher = final_mini_df['95_higher'].round().astype(int)
            round_90_lower = final_mini_df['90_lower'].round().astype(int)
            round_90_higher = final_mini_df['90_higher'].round().astype(int)
            round_80_lower = final_mini_df['80_lower'].round().astype(int)
            round_80_higher = final_mini_df['80_higher'].round().astype(int)
            rate_95 = sum((final_mini_df['test_y_true'] >= round_95_lower) & (
                    final_mini_df['test_y_true'] <= round_95_higher)) / len(final_mini_df)
            rate_90 = sum((final_mini_df['test_y_true'] >= round_90_lower) & (
                    final_mini_df['test_y_true'] <= round_90_higher)) / len(final_mini_df)
            rate_80 = sum((final_mini_df['test_y_true'] >= round_80_lower) & (
                    final_mini_df['test_y_true'] <= round_80_higher)) / len(final_mini_df)
        else:
            rate_95, rate_90, rate_80 = 0, 0, 0

        rate_95_lst.append(rate_95)
        rate_90_lst.append(rate_90)
        rate_80_lst.append(rate_80)

    return rate_95_lst, rate_90_lst, rate_80_lst


def distribution_func(dataset_name, b_lst, td_range, data_dicts, corr_name, data_arrange):
    feature_names = data_dicts["features"]
    num_features = len(feature_names)
    all_delta_lst = []

    for j in range(num_features):
        delta_lst = []
        for i in range(constants.BOOTSTRAP_NUMS):
            d = b_lst[0]["bootstrap_process"][i]["delta"]
            d_s = (d - td_range["delta"][0]) / (td_range["delta"][1] - td_range["delta"][0])
            delta = d_s[j]
            delta_lst.append(delta)
        all_delta_lst.append(delta_lst)

    all_delta_lst = np.array(all_delta_lst).flatten()
    delta_df = pd.DataFrame({"deltas": all_delta_lst})
    if dataset_name in ["occupancy", "pump_sensor", "air_quality", "power_demand"]:
        feature_values = []
        for i in range(num_features):
            feature_values.extend([f"Feature{i} {data_arrange}"] * 100)
        delta_df["features"] = feature_values
        delta_df['Method'] = corr_name
    else:
        raise RuntimeError(f"cannot find dataset_name: {dataset_name}")

    return delta_df


def tsmb_select_b_func(test_df_lst, B, y_test, task_name):
    y_test_copy = y_test.copy()
    final_indicator_lst = []

    for run_id in range(constants.BOOTSTRAP_ITERATIONS):
        pred_lst = []
        for i in range(B):
            if task_name == enums.TaskType.CLASSIFICATION.value:
                pred = test_df_lst[run_id][f'test_y_pred_proba1_{i}']
            elif task_name == enums.TaskType.REGRESSION.value:
                pred = test_df_lst[run_id][f'test_y_pred_{i}']
            else:
                raise ValueError(f'Invalid task_name: {task_name}')

            pred_lst.append(pred)

        pred_df = pd.DataFrame(pred_lst).T
        pred_mean_df = pd.DataFrame({
            'time': test_df_lst[run_id].iloc[:, 0],
            'y_pred': list(pred_df.iloc[:, :].mean(axis=1))
        })
        pred_mean_df = pred_mean_df.sort_values('time').dropna()
        y_test_copy.columns = ['time', 'y_true']
        final_df = pd.merge(y_test_copy, pred_mean_df, on='time', how='left').dropna()

        if task_name == enums.TaskType.CLASSIFICATION.value:
            final_auc = metrics.roc_auc_score(y_true=final_df.iloc[:, 1], y_score=final_df.iloc[:, 2])
            final_indicator_lst.append(final_auc)
        elif task_name == enums.TaskType.REGRESSION.value:
            final_r2 = r2_score(y_true=final_df.iloc[:, 1], y_pred=final_df.iloc[:, -1])
            final_indicator_lst.append(final_r2)

    return final_indicator_lst


def tdb_select_b_func(b_lst, B, X_training, y_training, X_test, y_test, task_name, tolerance):
    final_indicator_lst = []

    for run_id in range(constants.BOOTSTRAP_ITERATIONS):
        d_lst = [b_lst[run_id]['bootstrap_process'][j]['delta'] for j in np.arange(0, B)]
        w_lst = [b_lst[run_id]['bootstrap_process'][j]['window_size'] for j in np.arange(0, B)]
        final_d = np.array(d_lst).mean(axis=0)
        final_w = np.array(w_lst).mean(axis=0)
        final_data, final_data_test = opt_utils.transform_data(delta=final_d, window_size=final_w, X_training=X_training,
                                                   y_training=y_training, X_test=X_test, y_test=y_test, tolerance=tolerance)

        if task_name == enums.TaskType.CLASSIFICATION.value:
            final_rfc = GradientBoostingClassifier(random_state=seed)
            final_rfc.fit(final_data.iloc[:, 1:-1], final_data.iloc[:, -1])
            final_y_pred = final_rfc.predict(final_data_test.iloc[:, 1:-1])
            final_fpr, final_tpr, final_thresholds = metrics.roc_curve(final_data_test.iloc[:, -1], final_y_pred)
            final_auc = metrics.auc(final_fpr, final_tpr)
            final_indicator_lst.append(final_auc)

        elif task_name == enums.TaskType.REGRESSION.value:
            final_forest = GradientBoostingRegressor(random_state=seed)
            final_forest.fit(final_data.iloc[:, 1:-1], final_data.iloc[:, -1])
            final_test_predict = final_forest.predict(final_data_test.iloc[:, 1:-1])
            final_test_score = r2_score(final_data_test.iloc[:, -1], final_test_predict)
            final_indicator_lst.append(final_test_score)

        else:
            raise RuntimeError(f'cannot find task_name: {task_name}')

    return final_indicator_lst


def tft_func(tft_df_lst, task_name):
    indicator_lst = []

    for j in range(len(tft_df_lst)):
        y_pred_col = []
        for i in tft_df_lst[j].columns:
            if i.startswith('y_pred'):
                y_pred_col.append(i)
        tft_df_lst[j]['pred_mean'] = tft_df_lst[j][y_pred_col].mean(axis=1)
        tft_df_lst[j].dropna(inplace=True)

        if task_name == enums.TaskType.CLASSIFICATION.value:
            fpr, tpr, thresholds = metrics.roc_curve(tft_df_lst[j].iloc[:, 1], tft_df_lst[j].iloc[:, -1])
            indicator = metrics.auc(fpr, tpr)
        elif task_name == enums.TaskType.REGRESSION.value:
            indicator = r2_score(y_true=tft_df_lst[j].iloc[:, 1], y_pred=tft_df_lst[j].iloc[:, -1])
        else: indicator = None

        indicator_lst.append(indicator)

    return indicator_lst


def tft_select_b_func(test_df_lst, B, y_test, task_name):
    y_test_copy = y_test.copy()
    final_indicator_lst = []

    for run_id in range(constants.BOOTSTRAP_ITERATIONS):
        pred_lst = []
        for i in range(B):
            pred = test_df_lst[run_id][f'y_pred_{i}']
            pred_lst.append(pred)
        pred_df = (pd.DataFrame(pred_lst)).T
        pred_mean_df = pd.DataFrame(
            {'time': test_df_lst[run_id].iloc[:, 0], 'y_pred': list(pred_df.iloc[:, :].mean(axis=1))})

        if task_name == enums.TaskType.CLASSIFICATION.value:
            pred_mean_df['final_y_pred'] = (pred_mean_df['y_pred'] > 0.5).astype(int)
            pred_mean_df = pred_mean_df.sort_values([pred_mean_df.columns[0]])
            pred_mean_df.loc[pred_mean_df['y_pred'].isna(), 'final_y_pred'] = np.nan
            y_test_copy.columns = ['time', 'y_true']
            final_df = pd.merge(y_test_copy, pred_mean_df, on='time', how='left')
            final_df.dropna(inplace=True)
            indicator = metrics.roc_auc_score(y_true=final_df.iloc[:, 1], y_score=final_df.iloc[:, 2])
        elif task_name == enums.TaskType.REGRESSION.value:
            pred_mean_df = pred_mean_df.sort_values([pred_mean_df.columns[0]])
            pred_mean_df.dropna(inplace=True)
            y_test_copy.columns = ['time', 'y_true']
            final_df = pd.merge(y_test_copy, pred_mean_df, on='time', how='left')
            final_df.dropna(inplace=True)
            indicator = r2_score(y_true=final_df.iloc[:, 1], y_pred=final_df.iloc[:, -1])
        else: indicator=None

        final_indicator_lst.append(indicator)

    return final_indicator_lst


def make_predict_df(delta, window_size, X_training, y_training, X_test, y_test, method_name, task_name, tolerance):
    data, data_test = opt_utils.transform_data(delta=delta, window_size=window_size, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, tolerance=tolerance)

    if task_name == enums.TaskType.CLASSIFICATION.value:
        model = GradientBoostingClassifier(random_state=seed)
        target_column = f'{method_name}_y_pred_proba1'
        target_data = data_test.iloc[:, 1:-1]
    elif task_name == enums.TaskType.REGRESSION.value:
        model = GradientBoostingRegressor(random_state=seed)
        target_column = f'{method_name}_y_pred'
        target_data = data_test.iloc[:, 1:-1]
    else:
        raise ValueError("Invalid task_name value. Expected 'classification' or 'regression'.")

    model.fit(data.iloc[:, 1:-1], data.iloc[:, -1])
    predictions = model.predict(target_data)
    df = pd.DataFrame({'time': data_test.iloc[:, 0], target_column: predictions.T})

    return df


def make_confidence_df(cc_95, cc_90, cc_80, mi_95, mi_90, mi_80, dataset_name):
    final_bootstrap_df = pd.DataFrame()
    final_bootstrap_df['method'] = (['80%'] * 5 + ['90%'] * 5 + ['95%'] * 5) * 2
    final_bootstrap_df['coverage'] = [mi_80[0]] + [mi_80[1]] + [mi_80[2]] + [mi_80[3]] + [mi_80[4]] + \
                                     [mi_90[0]] + [mi_90[1]] + [mi_90[2]] + [mi_90[3]] + [mi_90[4]] + \
                                     [mi_95[0]] + [mi_95[1]] + [mi_95[2]] + [mi_95[3]] + [mi_95[4]] + \
                                     [cc_80[0]] + [cc_80[1]] + [cc_80[2]] + [cc_80[3]] + [cc_80[4]] + \
                                     [cc_90[0]] + [cc_90[1]] + [cc_90[2]] + [cc_90[3]] + [cc_90[4]] + \
                                     [cc_95[0]] + [cc_95[1]] + [cc_95[2]] + [cc_95[3]] + [cc_95[4]]

    final_bootstrap_df['run_id'] = (['test_0'] + ['test_1'] + ['test_2'] + ['test_3'] + ['test_4']) * 6
    final_bootstrap_df['Method name'] = ['TSMB-TDMI 80%'] * 5 + ['TSMB-TDMI 90%'] * 5 + ['TSMB-TDMI 95%'] * 5 + [
        'TSMB-GCC 80%'] * 5 + ['TSMB-GCC 90%'] * 5 + ['TSMB-GCC 95%'] * 5
    final_bootstrap_df['Dataset'] = dataset_name

    return final_bootstrap_df


def make_top_df(all_cc_top_lst, all_mi_top_lst, method_type, dataset_name):
    final_bootstrap_df = pd.DataFrame()
    final_bootstrap_df['method'] = (['top100'] * 5 + ['top50'] * 5 + ['top20'] * 5 + ['top10'] * 5 + ['top5'] * 5 + ['top1'] * 5) * 2

    final_bootstrap_df['test_values'] = [all_mi_top_lst[5][0]] + [all_mi_top_lst[5][1]] + [all_mi_top_lst[5][2]] + [all_mi_top_lst[5][3]] + [all_mi_top_lst[5][4]] + \
                                        [all_mi_top_lst[4][0]] + [all_mi_top_lst[4][1]] + [all_mi_top_lst[4][2]] + [all_mi_top_lst[4][3]] + [all_mi_top_lst[4][4]] + \
                                        [all_mi_top_lst[3][0]] + [all_mi_top_lst[3][1]] + [all_mi_top_lst[3][2]] + [all_mi_top_lst[3][3]] + [all_mi_top_lst[3][4]] + \
                                        [all_mi_top_lst[2][0]] + [all_mi_top_lst[2][1]] + [all_mi_top_lst[2][2]] + [all_mi_top_lst[2][3]] + [all_mi_top_lst[2][4]] + \
                                        [all_mi_top_lst[1][0]] + [all_mi_top_lst[1][1]] + [all_mi_top_lst[1][2]] + [all_mi_top_lst[1][3]] + [all_mi_top_lst[1][4]] + \
                                        [all_mi_top_lst[0][0]] + [all_mi_top_lst[0][1]] + [all_mi_top_lst[0][2]] + [all_mi_top_lst[0][3]] + [all_mi_top_lst[0][4]] + \
                                        [all_cc_top_lst[5][0]] + [all_cc_top_lst[5][1]] + [all_cc_top_lst[5][2]] + [all_cc_top_lst[5][3]] + [all_cc_top_lst[5][4]] + \
                                        [all_cc_top_lst[4][0]] + [all_cc_top_lst[4][1]] + [all_cc_top_lst[4][2]] + [all_cc_top_lst[4][3]] + [all_cc_top_lst[4][4]] + \
                                        [all_cc_top_lst[3][0]] + [all_cc_top_lst[3][1]] + [all_cc_top_lst[3][2]] + [all_cc_top_lst[3][3]] + [all_cc_top_lst[3][4]] + \
                                        [all_cc_top_lst[2][0]] + [all_cc_top_lst[2][1]] + [all_cc_top_lst[2][2]] + [all_cc_top_lst[2][3]] + [all_cc_top_lst[2][4]] + \
                                        [all_cc_top_lst[1][0]] + [all_cc_top_lst[1][1]] + [all_cc_top_lst[1][2]] + [all_cc_top_lst[1][3]] + [all_cc_top_lst[1][4]] + \
                                        [all_cc_top_lst[0][0]] + [all_cc_top_lst[0][1]] + [all_cc_top_lst[0][2]] + [all_cc_top_lst[0][3]] + [all_cc_top_lst[0][4]]

    final_bootstrap_df['run_id'] = (['test_0'] + ['test_1'] + ['test_2'] + ['test_3'] + ['test_4']) * 12
    final_bootstrap_df['Method name'] = [f'{method_type}-TDMI top100'] * 5 + [f'{method_type}-TDMI top50'] * 5 + [f'{method_type}-TDMI top20'] * 5 + [f'{method_type}-TDMI top10'] * 5 + \
                                        [f'{method_type}-TDMI top5'] * 5 + [f'{method_type}-TDMI top1'] * 5 + \
                                        [f'{method_type}-GCC top100'] * 5 + [f'{method_type}-GCC top50'] * 5 + [f'{method_type}-GCC top20'] * 5 + [f'{method_type}-GCC top10'] * 5 + \
                                        [f'{method_type}-GCC top5'] * 5 + [f'{method_type}-GCC top1'] * 5
    final_bootstrap_df['Dataset'] = dataset_name

    return final_bootstrap_df


def make_select_b_df(all_final_cc_lst, all_final_mi_lst, method_name, dataset_name):
    final_bootstrap_df = pd.DataFrame()
    final_bootstrap_df['method'] = (['100'] * 5 + ['50'] * 5 + ['20'] * 5 + ['10'] * 5 + ['5'] * 5 + ['1'] * 5) * 2

    final_bootstrap_df['test_values'] = [all_final_mi_lst[5][0]] + [all_final_mi_lst[5][1]] + [all_final_mi_lst[5][2]] + [all_final_mi_lst[5][3]] + [all_final_mi_lst[5][4]] + \
                                        [all_final_mi_lst[4][0]] + [all_final_mi_lst[4][1]] + [all_final_mi_lst[4][2]] + [all_final_mi_lst[4][3]] + [all_final_mi_lst[4][4]] + \
                                        [all_final_mi_lst[3][0]] + [all_final_mi_lst[3][1]] + [all_final_mi_lst[3][2]] + [all_final_mi_lst[3][3]] + [all_final_mi_lst[3][4]] + \
                                        [all_final_mi_lst[2][0]] + [all_final_mi_lst[2][1]] + [all_final_mi_lst[2][2]] + [all_final_mi_lst[2][3]] + [all_final_mi_lst[2][4]] + \
                                        [all_final_mi_lst[1][0]] + [all_final_mi_lst[1][1]] + [all_final_mi_lst[1][2]] + [all_final_mi_lst[1][3]] + [all_final_mi_lst[1][4]] + \
                                        [all_final_mi_lst[0][0]] + [all_final_mi_lst[0][1]] + [all_final_mi_lst[0][2]] + [all_final_mi_lst[0][3]] + [all_final_mi_lst[0][4]] + \
                                        [all_final_cc_lst[5][0]] + [all_final_cc_lst[5][1]] + [all_final_cc_lst[5][2]] + [all_final_cc_lst[5][3]] + [all_final_cc_lst[5][4]] + \
                                        [all_final_cc_lst[4][0]] + [all_final_cc_lst[4][1]] + [all_final_cc_lst[4][2]] + [all_final_cc_lst[4][3]] + [all_final_cc_lst[4][4]] + \
                                        [all_final_cc_lst[3][0]] + [all_final_cc_lst[3][1]] + [all_final_cc_lst[3][2]] + [all_final_cc_lst[3][3]] + [all_final_cc_lst[3][4]] + \
                                        [all_final_cc_lst[2][0]] + [all_final_cc_lst[2][1]] + [all_final_cc_lst[2][2]] + [all_final_cc_lst[2][3]] + [all_final_cc_lst[2][4]] + \
                                        [all_final_cc_lst[1][0]] + [all_final_cc_lst[1][1]] + [all_final_cc_lst[1][2]] + [all_final_cc_lst[1][3]] + [all_final_cc_lst[1][4]] + \
                                        [all_final_cc_lst[0][0]] + [all_final_cc_lst[0][1]] + [all_final_cc_lst[0][2]] + [all_final_cc_lst[0][3]] + [all_final_cc_lst[0][4]]

    final_bootstrap_df['run_id'] = (['test_0'] + ['test_1'] + ['test_2'] + ['test_3'] + ['test_4']) * 12
    final_bootstrap_df['Method name'] = [f'{method_name}-TDMI B100'] * 5 + [f'{method_name}-TDMI B50'] * 5 + [f'{method_name}-TDMI B20'] * 5 + [f'{method_name}-TDMI B10'] * 5 + \
                                        [f'{method_name}-TDMI B5'] * 5 + [f'{method_name}-TDMI B1'] * 5 + \
                                        [f'{method_name}-GCC B100'] * 5 + [f'{method_name}-GCC B50'] * 5 + [f'{method_name}-GCC B20'] * 5 + [f'{method_name}-GCC B10'] * 5 + \
                                        [f'{method_name}-GCC B5'] * 5 + [f'{method_name}-GCC B1'] * 5
    final_bootstrap_df['Dataset'] = dataset_name

    return final_bootstrap_df


def make_orderly_df(orderly_dict, dataset_name):
    if dataset_name in ['air_quality', 'pump_sensor', 'occupancy', 'power_demand']:
        orderly_df = pd.DataFrame()
        orderly_df['Method'] = ['GCC', 'TDMI', 'No_alignment']
        orderly_df['test_values'] = [orderly_dict['orderly_test_result_dict']['orderly_cc_indicator']] + \
                                    [orderly_dict['orderly_test_result_dict']['orderly_mi_indicator']] + \
                                    [orderly_dict['orderly_test_result_dict']['orderly_no_alignment_indicator']]

        orderly_df['run_id'] = ['test_0'] * 3

        return orderly_df
    else:
        raise RuntimeError(f'cannot find dataset_name: {dataset_name}')



def make_tsmb_df(cc_tsmb_performance_lst, mi_tsmb_performance_lst):
    final_bootstrap_df = pd.DataFrame()
    final_bootstrap_df['Method'] = ['TSMB-TDMI'] * 5 + ['TSMB-GCC'] * 5

    final_bootstrap_df['test_values'] = [mi_tsmb_performance_lst[0]] + [mi_tsmb_performance_lst[1]] + [mi_tsmb_performance_lst[2]] + \
                                        [mi_tsmb_performance_lst[3]] + [mi_tsmb_performance_lst[4]] + \
                                        [cc_tsmb_performance_lst[0]] + [cc_tsmb_performance_lst[1]] + [cc_tsmb_performance_lst[2]] + \
                                        [cc_tsmb_performance_lst[3]] + [cc_tsmb_performance_lst[4]]

    final_bootstrap_df['run_id'] = ['test_0', 'test_1', 'test_2', 'test_3', 'test_4'] * 2

    return final_bootstrap_df


def make_perturbed_df(cc_perturbed_performance_lst, mi_perturbed_performancelst):
    final_bootstrap_df = pd.DataFrame()
    final_bootstrap_df['Method'] = ['Perturbed Model-TDMI']*5 + ['Perturbed Model-GCC']*5

    final_bootstrap_df['test_values'] = [mi_perturbed_performancelst[0]] + [mi_perturbed_performancelst[1]] + [mi_perturbed_performancelst[2]] + \
                                        [mi_perturbed_performancelst[3]] + [mi_perturbed_performancelst[4]] + \
                                        [cc_perturbed_performance_lst[0]] + [cc_perturbed_performance_lst[1]] + [cc_perturbed_performance_lst[2]] + \
                                        [cc_perturbed_performance_lst[3]] + [cc_perturbed_performance_lst[4]]

    final_bootstrap_df['run_id'] = ['test_0', 'test_1', 'test_2', 'test_3', 'test_4'] * 2

    return final_bootstrap_df


def make_tdb_df(cc_tdb_performance_lst, mi_tdb_performance_lst):
    final_bootstrap_df = pd.DataFrame()
    final_bootstrap_df['Method'] = ['TDB-TDMI'] * 5  + ['TDB-GCC'] * 5

    final_bootstrap_df['test_values'] = [mi_tdb_performance_lst[0]] + [mi_tdb_performance_lst[1]] + [mi_tdb_performance_lst[2]] + [mi_tdb_performance_lst[3]] + [mi_tdb_performance_lst[4]] + \
                                        [cc_tdb_performance_lst[0]] + [cc_tdb_performance_lst[1]] + [cc_tdb_performance_lst[2]] + [cc_tdb_performance_lst[3]] + [cc_tdb_performance_lst[4]]

    final_bootstrap_df['run_id'] = (['test_0'] + ['test_1'] + ['test_2'] + ['test_3'] + ['test_4']) * 2

    return final_bootstrap_df


def make_performance_df(orderly_dict, dataset_name, cc_tsmb_performance_lst, mi_tsmb_performance_lst, cc_perturbed_performance_lst,
                  mi_perturbed_performancelst, cc_tdb_performance_lst, mi_tdb_performance_lst):
    orderly_df = make_orderly_df(orderly_dict=orderly_dict, dataset_name=dataset_name)
    tsmb_df = make_tsmb_df(cc_tsmb_performance_lst=cc_tsmb_performance_lst, mi_tsmb_performance_lst=mi_tsmb_performance_lst)
    perturbed_df = make_perturbed_df(cc_perturbed_performance_lst=cc_perturbed_performance_lst, mi_perturbed_performancelst=mi_perturbed_performancelst)
    tdb_df = make_tdb_df(cc_tdb_performance_lst=cc_tdb_performance_lst, mi_tdb_performance_lst=mi_tdb_performance_lst)

    final_df = pd.concat([orderly_df, tsmb_df, perturbed_df, tdb_df], axis=0)
    final_df['Dataset'] = dataset_name
    final_df.reset_index(drop=True, inplace=True)

    return final_df


def make_tft_performance_df(tft_orderly_dict, cc_tft_lst, mi_tft_lst, dataset_name):
    tft_df = pd.DataFrame({'Method': ['TSMB-TDMI', 'TSMB-TDMI', 'TSMB-TDMI', 'TSMB-TDMI', 'TSMB-TDMI',
                                        'TSMB-GCC', 'TSMB-GCC', 'TSMB-GCC', 'TSMB-GCC', 'TSMB-GCC', 'TDMI', 'GCC'],
                           'test_values':mi_tft_lst + cc_tft_lst + [tft_orderly_dict['orderly_tft_mi_indicator']] + [tft_orderly_dict['orderly_tft_cc_indicator']],
                           'run_id': ['test_0', 'test_1', 'test_2', 'test_3', 'test_4', 'test_0', 'test_1', 'test_2', 'test_3', 'test_4', 'test_0', 'test_0'],
                           'Dataset': [dataset_name] * 12})
    # tft_df['test_values'] = tft_df['test_values'] - tft_orderly_dict['orderly_tft_no_alignment_indicator']

    return tft_df


def update_dataset_names(dataset_names, mapping):
    updated_names = [mapping.get(name, name) for name in dataset_names]
    return updated_names


def merge_results(*dataset_names):
    tsmb_top_lst = []
    perturbed_top_lst = []
    tdb_top_lst = []
    tsmb_select_b_lst = []
    perturbed_select_b_lst = []
    tdb_select_b_lst = []
    confidence_tsmb_lst = []
    all_performance_lst = []
    performance_lst = []
    tft_performance_lst = []
    tft_select_b_lst = []

    mapping = {
        enums.DataSetName.OCCUPANCY.value: 'Occupancy',
        enums.DataSetName.PUMP_SENSOR.value: 'Water Pump',
        enums.DataSetName.POWER_DEMAND.value: 'Power Demand',
        enums.DataSetName.AIR_QUALITY.value: 'Air Quality'
    }
    updated_names = update_dataset_names(dataset_names, mapping)

    for name, updated_name in zip(dataset_names, updated_names):
        if name != 'mineral_processing':
            fixed_tsmb_top = pd.read_csv(f'{name}_fixed_tsmb_top.csv')
            fixed_perturbed_top = pd.read_csv(f'{name}_fixed_perturbed_top.csv')
            fixed_tdb_top = pd.read_csv(f'{name}_fixed_tdb_top.csv')
            fixed_tsmb_select_b = pd.read_csv(f'{name}_fixed_gbdt_tsmb_select_b.csv')
            fixed_perturbed_select_b = pd.read_csv(f'{name}_fixed_gbdt_perturbed_select_b.csv')
            fixed_tdb_select_b = pd.read_csv(f'{name}_fixed_gbdt_tdb_select_b.csv')
            fixed_confidence_tsmb = pd.read_csv(f'{name}_fixed_tsmb_confidence.csv')
            fixed_all_performance = pd.read_csv(f'{name}_fixed_tsmb_all_performance.csv')
            fixed_performance = pd.read_csv(f'{name}_fixed_tsmb_performance.csv')
            fixed_tft_performance = pd.read_csv(f'{name}_fixed_tft_tsmb_performance.csv')
            fixed_tft_select_b = pd.read_csv(f'{name}_fixed_tft_select_b.csv')
            fixed_tsmb_top['Dataset'] = updated_name + ' Fixed'
            fixed_perturbed_top['Dataset'] = updated_name + ' Fixed'
            fixed_tdb_top['Dataset'] = updated_name + ' Fixed'
            fixed_tsmb_select_b['Dataset'] = updated_name + ' Fixed'
            fixed_perturbed_select_b['Dataset'] = updated_name + ' Fixed'
            fixed_tdb_select_b['Dataset'] = updated_name + ' Fixed'
            fixed_confidence_tsmb['Dataset'] = updated_name + ' Fixed'
            fixed_all_performance['Dataset'] = updated_name + ' Fixed'
            fixed_performance['Dataset'] = updated_name + ' Fixed'
            fixed_tft_performance['Dataset'] = updated_name + ' Fixed'
            fixed_tft_select_b['Dataset'] = updated_name + ' Fixed'

            stochastic_tsmb_top = pd.read_csv(f'{name}_stochastic_tsmb_top.csv')
            stochastic_perturbed_top = pd.read_csv(f'{name}_stochastic_perturbed_top.csv')
            stochastic_tdb_top = pd.read_csv(f'{name}_stochastic_tdb_top.csv')
            stochastic_tsmb_select_b = pd.read_csv(f'{name}_stochastic_gbdt_tsmb_select_b.csv')
            stochastic_perturbed_select_b = pd.read_csv(f'{name}_stochastic_gbdt_perturbed_select_b.csv')
            stochastic_tdb_select_b = pd.read_csv(f'{name}_stochastic_gbdt_tdb_select_b.csv')
            stochastic_confidence_tsmb = pd.read_csv(f'{name}_stochastic_tsmb_confidence.csv')
            stochastic_all_performance = pd.read_csv(f'{name}_stochastic_tsmb_all_performance.csv')
            stochastic_performance = pd.read_csv(f'{name}_stochastic_tsmb_performance.csv')
            stochastic_tft_performance = pd.read_csv(f'{name}_stochastic_tft_tsmb_performance.csv')
            stochastic_tft_select_b = pd.read_csv(f'{name}_stochastic_tft_select_b.csv')
            stochastic_tsmb_top['Dataset'] = updated_name + ' Stochastic'
            stochastic_perturbed_top['Dataset'] = updated_name + ' Stochastic'
            stochastic_tdb_top['Dataset'] = updated_name + ' Stochastic'
            stochastic_tsmb_select_b['Dataset'] = updated_name + ' Stochastic'
            stochastic_perturbed_select_b['Dataset'] = updated_name + ' Stochastic'
            stochastic_tdb_select_b['Dataset'] = updated_name + ' Stochastic'
            stochastic_confidence_tsmb['Dataset'] = updated_name + ' Stochastic'
            stochastic_all_performance['Dataset'] = updated_name + ' Stochastic'
            stochastic_performance['Dataset'] = updated_name + ' Stochastic'
            stochastic_tft_performance['Dataset'] = updated_name + ' Stochastic'
            stochastic_tft_select_b['Dataset'] = updated_name + ' Stochastic'


            fixed_concat_tsmb_top = pd.concat([fixed_tsmb_top, stochastic_tsmb_top])
            fixed_concat_perturbed_top = pd.concat([fixed_perturbed_top, stochastic_perturbed_top])
            fixed_concat_tdb_top = pd.concat([fixed_tdb_top, stochastic_tdb_top])
            fixed_concat_tsmb_select_b = pd.concat([fixed_tsmb_select_b, stochastic_tsmb_select_b])
            fixed_concat_perturbed_select_b = pd.concat([fixed_perturbed_select_b, stochastic_perturbed_select_b])
            fixed_concat_tdb_select_b = pd.concat([fixed_tdb_select_b, stochastic_tdb_select_b])
            fixed_concat_confidence_tsmb = pd.concat([fixed_confidence_tsmb, stochastic_confidence_tsmb])
            fixed_concat_all_performance = pd.concat([fixed_all_performance, stochastic_all_performance])
            fixed_concat_performance = pd.concat([fixed_performance, stochastic_performance])
            fixed_concat_tft_performance = pd.concat([fixed_tft_performance, stochastic_tft_performance])
            fixed_concat_tft_select_b = pd.concat([fixed_tft_select_b, stochastic_tft_select_b])

            tsmb_top_lst.append(fixed_concat_tsmb_top)
            perturbed_top_lst.append(fixed_concat_perturbed_top)
            tdb_top_lst.append(fixed_concat_tdb_top)
            tsmb_select_b_lst.append(fixed_concat_tsmb_select_b)
            perturbed_select_b_lst.append(fixed_concat_perturbed_select_b)
            tdb_select_b_lst.append(fixed_concat_tdb_select_b)
            confidence_tsmb_lst.append(fixed_concat_confidence_tsmb)
            all_performance_lst.append(fixed_concat_all_performance)
            performance_lst.append(fixed_concat_performance)
            tft_performance_lst.append(fixed_concat_tft_performance)
            tft_select_b_lst.append(fixed_concat_tft_select_b)

        elif name == 'mineral_processing':
            fixed_tsmb_top = pd.read_csv(f'{name}_fixed_tsmb_top.csv')
            fixed_perturbed_top = pd.read_csv(f'{name}_fixed_perturbed_top.csv')
            fixed_tdb_top = pd.read_csv(f'{name}_fixed_tdb_top.csv')
            fixed_tsmb_select_b = pd.read_csv(f'{name}_fixed_gbdt_tsmb_select_b.csv')
            fixed_perturbed_select_b = pd.read_csv(f'{name}_fixed_gbdt_perturbed_select_b.csv')
            fixed_tdb_select_b = pd.read_csv(f'{name}_fixed_gbdt_tdb_select_b.csv')
            fixed_confidence_tsmb = pd.read_csv(f'{name}_fixed_tsmb_confidence.csv')
            fixed_all_performance = pd.read_csv(f'{name}_fixed_tsmb_all_performance.csv')
            fixed_performance = pd.read_csv(f'{name}_fixed_tsmb_performance.csv')
            fixed_tft_performance = pd.read_csv(f'{name}_fixed_tft_tsmb_performance.csv')
            fixed_tft_select_b = pd.read_csv(f'{name}_fixed_tft_select_b.csv')
            fixed_tsmb_top['Dataset'] = updated_name
            fixed_perturbed_top['Dataset'] = updated_name
            fixed_tdb_top['Dataset'] = updated_name
            fixed_tsmb_select_b['Dataset'] = updated_name
            fixed_perturbed_select_b['Dataset'] = updated_name
            fixed_tdb_select_b['Dataset'] = updated_name
            fixed_confidence_tsmb['Dataset'] = updated_name
            fixed_all_performance['Dataset'] = updated_name
            fixed_performance['Dataset'] = updated_name
            fixed_tft_performance['Dataset'] = updated_name
            fixed_tft_select_b['Dataset'] = updated_name

            tsmb_top_lst.append(fixed_tsmb_top)
            perturbed_top_lst.append(fixed_perturbed_top)
            tdb_top_lst.append(fixed_tdb_top)
            tsmb_select_b_lst.append(fixed_tsmb_select_b)
            perturbed_select_b_lst.append(fixed_perturbed_select_b)
            tdb_select_b_lst.append(fixed_tdb_select_b)
            confidence_tsmb_lst.append(fixed_confidence_tsmb)
            all_performance_lst.append(fixed_all_performance)
            performance_lst.append(fixed_performance)
            tft_performance_lst.append(fixed_tft_performance)
            tft_select_b_lst.append(fixed_tft_select_b)

        else:
            raise RuntimeError(f"cannot find dataset_name: {name}")


    tsmb_top = pd.concat(tsmb_top_lst)
    perturbed_top = pd.concat(perturbed_top_lst)
    tdb_top = pd.concat(tdb_top_lst)
    tsmb_select_b = pd.concat(tsmb_select_b_lst)
    perturbed_select_b = pd.concat(perturbed_select_b_lst)
    tdb_select_b = pd.concat(tdb_select_b_lst)
    confidence_tsmb = pd.concat(confidence_tsmb_lst)
    all_performance = pd.concat(all_performance_lst)
    performance = pd.concat(performance_lst)
    tft_performance = pd.concat(tft_performance_lst)
    tft_select_b = pd.concat(tft_select_b_lst)

    return tsmb_top, perturbed_top, tdb_top, tsmb_select_b, perturbed_select_b, tdb_select_b, confidence_tsmb, all_performance, performance, tft_performance, tft_select_b



