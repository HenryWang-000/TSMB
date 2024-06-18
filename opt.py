import numpy as np
import pandas as pd
import fire
import stochastic_function
import constants
import opt_utils
import enums
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
from functools import reduce
from scipy.io import arff
from sklearn.model_selection import train_test_split

seed = constants.RAMDOM_SEED



def make_occupancy_df(data_arrange, file_path='occupancy_data.csv'):
    occupancy = pd.read_csv(file_path)
    occupancy.iloc[:, 0] = pd.to_datetime(occupancy.iloc[:, 0])
    occupancy.rename(columns={"date": "time"}, inplace=True)

    if data_arrange == enums.DataArrange.FIXED.value:
        df0 = occupancy[['time', 'Temperature']]
        df1 = occupancy[['time', 'Humidity']]
        df2 = occupancy[['time', 'Light']]
        df3 = occupancy[['time', 'CO2']]
        df4 = occupancy[['time', 'HumidityRatio']]
        df_lst = [df0, df1, df2, df3, df4]

        real_delta = [150, 120, 90, 60, 30]
        for i in range(5):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=real_delta[i])

        lable_df = occupancy[['time', 'Occupancy']]
        df_lst.append(lable_df)
        sorted_df = reduce(lambda left, right: pd.merge(left, right, on=['time']), df_lst)
    elif data_arrange == enums.DataArrange.STOCHASTIC.value:
        idx_df = stochastic_function.get_random_idx_df(occupancy)
        delta_lsts = [[180, 140, 100, 80, 50], [170, 130, 100, 70, 40], [150, 120, 90, 60, 30], [140, 110, 80, 70, 20], [130, 110, 80, 50, 20]]
        dd = stochastic_function.get_occupancy_df_lsts(delta_lsts=delta_lsts, df=idx_df)
        sorted_df = stochastic_function.get_sorted_df(df=idx_df, delta_df_lst=dd)
    else: sorted_df = pd.DataFrame()

    occupancy_X_df = sorted_df.iloc[:, :-1]
    occupancy_y_df = sorted_df[['time', 'Occupancy']]
    occupancy_X_training, occupancy_X_test, occupancy_y_training, occupancy_y_test = opt_utils.train_test_splits(X_df=occupancy_X_df, y_df=occupancy_y_df, test_rate=0.5, tolerance='1min')
    occupancy_X_train, occupancy_X_val, occupancy_y_train, occupancy_y_val = opt_utils.train_test_splits(X_df=occupancy_X_training, y_df=occupancy_y_training, test_rate=0.5, tolerance='1min')

    dicts = constants.OCCUPANCY_TD_RANGE

    return occupancy_X_training, occupancy_y_training, occupancy_X_train, occupancy_y_train, occupancy_X_test, occupancy_y_test, occupancy_X_val, occupancy_y_val, dicts



def make_sensor_df(data_arrange, file_path='sensor.csv'):
    sensor = pd.read_csv(file_path)
    sensor['machine_status'].replace('RECOVERING', 'BROKEN', inplace=True)
    sensor.rename(columns={"timestamp": "time"}, inplace=True)
    le = LabelEncoder()
    sensor.iloc[:, -1] = le.fit_transform(sensor.iloc[:, -1])
    sensor = sensor[['time', 'sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09', 'sensor_10', 'machine_status']]
    sensor.iloc[:, 0] = pd.to_datetime(sensor.iloc[:, 0])
    sensor_df = sensor.iloc[69000:80000, :]
    sensor_df = sensor_df.dropna(axis=0, how='any', subset=None, inplace=False)
    sensor_df.reset_index(drop=True, inplace=True)

    if data_arrange == enums.DataArrange.STOCHASTIC.value:
        idx_df = stochastic_function.get_random_idx_df(sensor_df)
        delta_lsts = [[75, 65, 50, 40, 30, 25], [70, 60, 45, 35, 25, 20], [65, 55, 40, 30, 20, 15], [60, 50, 35, 25, 15, 10], [55, 45, 30, 20, 10, 5]]
        dd = stochastic_function.get_pump_sensor_df_lsts(delta_lsts=delta_lsts, df=idx_df)
        sorted_df = stochastic_function.get_sorted_df(df=idx_df, delta_df_lst=dd)
    elif data_arrange == enums.DataArrange.FIXED.value:
        df0 = sensor_df[['time', 'sensor_04']]
        df1 = sensor_df[['time', 'sensor_06']]
        df2 = sensor_df[['time', 'sensor_07']]
        df3 = sensor_df[['time', 'sensor_08']]
        df4 = sensor_df[['time', 'sensor_09']]
        df5 = sensor_df[['time', 'sensor_10']]
        df_lst = [df0, df1, df2, df3, df4, df5]

        real_delta = [65, 55, 40, 30, 20, 15]
        for i in range(6):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=real_delta[i])

        lable_df = sensor_df[['time', 'machine_status']]
        df_lst.append(lable_df)
        sorted_df = reduce(lambda left, right: pd.merge(left, right, on=['time']), df_lst)
    else: sorted_df = pd.DataFrame()

    sensor_X_df = sorted_df.iloc[:, 0:-1]
    sensor_y_df = sorted_df[['time', 'machine_status']]

    sensor_X_train, sensor_X_val, sensor_y_train, sensor_y_val = opt_utils.train_test_splits(X_df=sensor_X_df, y_df=sensor_y_df, test_rate=0.5, tolerance='1min')

    dicts = constants.PUMP_SENSOR_TD_RANGE

    return sensor_X_df, sensor_y_df, sensor_X_train, sensor_y_train, sensor_X_df, sensor_y_df, sensor_X_val, sensor_y_val, dicts


def make_all_sensor_df(data_arrange, file_path='sensor.csv'):
    sensor = pd.read_csv(file_path)
    sensor['machine_status'].replace('RECOVERING', 'BROKEN', inplace=True)
    sensor.rename(columns={"timestamp": "time"}, inplace=True)
    le = LabelEncoder()
    sensor.iloc[:, -1] = le.fit_transform(sensor.iloc[:, -1])
    sensor = sensor[
        ['time', 'sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09', 'sensor_10', 'machine_status']]
    sensor.iloc[:, 0] = pd.to_datetime(sensor.iloc[:, 0])
    all_sensor_df = sensor.copy()
    all_sensor_df = all_sensor_df.dropna(axis=0, how='any', subset=None, inplace=False)
    all_sensor_df.reset_index(drop=True, inplace=True)

    if data_arrange == enums.DataArrange.STOCHASTIC.value:
        idx_df = stochastic_function.get_random_idx_df(all_sensor_df)
        delta_lsts = [[75, 65, 50, 40, 30, 25], [70, 60, 45, 35, 25, 20], [65, 55, 40, 30, 20, 15],
                      [60, 50, 35, 25, 15, 10], [55, 45, 30, 20, 10, 5]]
        dd = stochastic_function.get_pump_sensor_df_lsts(delta_lsts=delta_lsts, df=idx_df)
        sorted_df = stochastic_function.get_sorted_df(df=idx_df, delta_df_lst=dd)
    elif data_arrange == enums.DataArrange.FIXED.value:
        df0 = all_sensor_df[['time', 'sensor_04']]
        df1 = all_sensor_df[['time', 'sensor_06']]
        df2 = all_sensor_df[['time', 'sensor_07']]
        df3 = all_sensor_df[['time', 'sensor_08']]
        df4 = all_sensor_df[['time', 'sensor_09']]
        df5 = all_sensor_df[['time', 'sensor_10']]
        df_lst = [df0, df1, df2, df3, df4, df5]

        real_delta = [65, 55, 40, 30, 20, 15]
        for i in range(6):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=real_delta[i])

        lable_df = all_sensor_df[['time', 'machine_status']]
        df_lst.append(lable_df)
        sorted_df = reduce(lambda left, right: pd.merge(left, right, on=['time']), df_lst)
    else:
        sorted_df = pd.DataFrame()

    all_sensor_X_df = sorted_df.iloc[:, 0:-1]
    all_sensor_y_df = sorted_df[['time', 'machine_status']]

    all_sensor_X_training, all_sensor_X_test, all_sensor_y_training, all_sensor_y_test = opt_utils.train_test_splits(X_df=all_sensor_X_df,
                                                                                             y_df=all_sensor_y_df,
                                                                                             test_rate=0.5,
                                                                                             tolerance='1min')

    dicts = constants.PUMP_SENSOR_TD_RANGE

    return all_sensor_X_training, all_sensor_y_training, all_sensor_X_test, all_sensor_y_test, dicts



def make_air_df(data_arrange, file_path='air_data.csv'):
    air_df = pd.read_csv(file_path)
    air_df.loc[:, 'time'] = pd.to_datetime(air_df.loc[:, 'time'])
    air_df = air_df[['time', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'CO(GT)']]
    air_df.replace(-200, np.nan, inplace=True)
    air_df.dropna(subset=['CO(GT)'], inplace=True)
    column_means = air_df.mean()
    air_df = air_df.fillna(column_means).reset_index(drop=True)

    if data_arrange == enums.DataArrange.STOCHASTIC.value:
        idx_df = stochastic_function.get_random_idx_df(air_df)
        delta_lsts = [[1080, 1080, 780, 780, 480, 180, 180, 180], [1140, 1140, 840, 840, 540, 240, 240, 240],
                      [1200, 1200, 900, 900, 600, 300, 300, 300],
                      [1260, 1260, 960, 960, 660, 360, 360, 360], [1320, 1320, 1020, 1020, 720, 420, 420, 420]]
        dd = stochastic_function.get_air_quality_df_lsts(delta_lsts=delta_lsts, df=idx_df)
        sorted_df = stochastic_function.get_sorted_df(df=idx_df, delta_df_lst=dd)
    elif data_arrange == enums.DataArrange.FIXED.value:
        df0 = air_df[['time', 'PT08.S1(CO)']]
        df1 = air_df[['time', 'PT08.S2(NMHC)']]
        df2 = air_df[['time', 'PT08.S3(NOx)']]
        df3 = air_df[['time', 'PT08.S4(NO2)']]
        df4 = air_df[['time', 'PT08.S5(O3)']]
        df5 = air_df[['time', 'T']]
        df6 = air_df[['time', 'RH']]
        df7 = air_df[['time', 'AH']]
        df_lst = [df0, df1, df2, df3, df4, df5, df6, df7]

        real_delta = [1200, 1200, 900, 900, 600, 300, 300, 300]
        for i in range(8):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=real_delta[i])

        lable_df = air_df[['time', 'CO(GT)']]
        df_lst.append(lable_df)
        sorted_df = reduce(lambda left, right: pd.merge(left, right, on=['time']), df_lst)
    else: sorted_df = pd.DataFrame()

    air_X_df = sorted_df.iloc[:, 0:-1]
    air_y_df = sorted_df[['time', 'CO(GT)']]

    air_X_training, air_X_test, air_y_training, air_y_test = opt_utils.train_test_splits(X_df=air_X_df, y_df=air_y_df, test_rate=0.25, tolerance='12h')
    air_X_train, air_X_val, air_y_train, air_y_val = opt_utils.train_test_splits(X_df=air_X_training, y_df=air_y_training, test_rate=0.33, tolerance='12h')
    dicts = constants.AIR_QUALITY_TD_RANGE

    return air_X_training, air_y_training, air_X_train, air_y_train, air_X_test, air_y_test, air_X_val, air_y_val, dicts




def make_PowerDemand_df(data_arrange, train_file_path='ItalyPowerDemand_TRAIN.arff', test_file_path='ItalyPowerDemand_TEST.arff'):
    train_data, train_meta = arff.loadarff(train_file_path)
    test_data, test_meta = arff.loadarff(test_file_path)

    data = pd.DataFrame(train_data)
    data_test = pd.DataFrame(test_data)
    data['target'] = data['target'].map({b'1': 0, b'2': 1})
    data_test['target'] = data_test['target'].map({b'1': 0, b'2': 1})

    start_time = pd.Timestamp("2023-01-01 00:00:00")
    end_time = start_time + pd.DateOffset(minutes=len(data) - 1)
    time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
    data.insert(0, 'time', time_index)

    test_start_time = pd.Timestamp("2023-01-01 01:17:00")
    test_end_time = test_start_time + pd.DateOffset(minutes=len(data_test) - 1)
    test_time_index = pd.date_range(start=test_start_time, end=test_end_time, freq='1min')
    data_test.insert(0, 'time', test_time_index)

    if data_arrange == enums.DataArrange.STOCHASTIC.value:
        data_idx_df = stochastic_function.get_random_idx_df(data)
        data_test_idx_df = stochastic_function.get_random_idx_df(data_test)
        delta_lsts = [[9, 7, 5], [8, 6, 4], [7, 5, 3], [6, 4, 2], [5, 3, 1]]
        data_lst, data_test_lst = stochastic_function.get_power_demand_df_lsts(delta_lsts=delta_lsts, data=data_idx_df, data_test=data_test_idx_df)
        data_sorted_df = stochastic_function.get_sorted_df(df=data_idx_df, delta_df_lst=data_lst)
        data_test_sorted_df = stochastic_function.get_sorted_df(df=data_test_idx_df, delta_df_lst=data_test_lst)
        data_sorted_df, data_test_sorted_df = data_sorted_df.iloc[:, :-1], data_test_sorted_df.iloc[:, :-1]
    elif data_arrange == enums.DataArrange.FIXED.value:
        feature_groups = [
            data.columns[1:9],
            data.columns[9:17],
            data.columns[17:25]
        ]
        time_deltas = [7, 5, 3]

        data_dfs = [data, data_test]
        for df in data_dfs:
            for group, time_delta in zip(feature_groups, time_deltas):
                for column in group:
                    df[column] = df[column].shift(periods=-time_delta)

        data.dropna(inplace=True)
        data_test.dropna(inplace=True)
        data_sorted_df, data_test_sorted_df = data.copy(), data_test.copy()
    else: data_sorted_df, data_test_sorted_df = pd.DataFrame(), pd.DataFrame()

    powerDemand_train = data_sorted_df.copy()
    powerDemand_test = data_test_sorted_df.copy()
    powerDemand_train.iloc[:, 0] = pd.to_datetime(powerDemand_train.iloc[:, 0])
    powerDemand_test.iloc[:, 0] = pd.to_datetime(powerDemand_test.iloc[:, 0])

    power_X_training = powerDemand_train.iloc[:, :-1]
    power_y_training = powerDemand_train[['time', 'target']]

    power_X_test = powerDemand_test.iloc[:, :-1]
    power_y_test = powerDemand_test[['time', 'target']]

    dicts = constants.POWER_DEMAND_TD_RANGE

    return power_X_training, power_y_training, power_X_training, power_y_training, power_X_test, power_y_test, power_X_test, power_y_test, dicts


def make_model_df(data_set, data_arrange):
    if data_set == enums.DataSetName.OCCUPANCY.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = make_occupancy_df(file_path='occupancy_data.csv', data_arrange=data_arrange)

    elif data_set == enums.DataSetName.AIR_QUALITY.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = make_air_df(file_path='air_data.csv', data_arrange=data_arrange)

    elif data_set == enums.DataSetName.PUMP_SENSOR.value:
        part_X_training, part_y_training, part_X_train, part_y_train, part_X_test, part_y_test, part_X_val, part_y_val, dicts = make_sensor_df(file_path='sensor.csv', data_arrange=data_arrange)
        all_X_training, all_y_training, all_X_test, all_y_test, dicts = make_all_sensor_df(file_path='sensor.csv', data_arrange=data_arrange)
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = all_X_training, all_y_training, part_X_train, part_y_train, all_X_test, all_y_test, part_X_val, part_y_val, dicts

    elif data_set == enums.DataSetName.POWER_DEMAND.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val,  dicts = make_PowerDemand_df(train_file_path='ItalyPowerDemand_TRAIN.arff', test_file_path='ItalyPowerDemand_TEST.arff', data_arrange=data_arrange)

    else:
        raise RuntimeError(f'cannot find data_set: {data_set}')

    return X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts


def make_origin_df(data_set):
    if data_set == enums.DataSetName.OCCUPANCY.value:
        origin_df = pd.read_csv('occupancy_data.csv')
        ori_train, ori_test = train_test_split(origin_df, test_size=0.5, random_state=seed, shuffle=False)
    elif data_set == enums.DataSetName.AIR_QUALITY.value:
        origin_df = pd.read_csv('air_data.csv')
        cols = origin_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        origin_df = origin_df[cols]
        ori_train, ori_test = train_test_split(origin_df, test_size=0.25, random_state=seed, shuffle=False)
    elif data_set == enums.DataSetName.PUMP_SENSOR.value:
        sensor = pd.read_csv('sensor.csv')
        sensor['machine_status'].replace('RECOVERING', 'BROKEN', inplace=True)
        sensor.rename(columns={"timestamp": "time"}, inplace=True)
        le = LabelEncoder()
        sensor.iloc[:, -1] = le.fit_transform(sensor.iloc[:, -1])
        sensor = sensor[
            ['time', 'sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09', 'sensor_10', 'machine_status']]
        sensor = sensor.dropna(axis=0, how='any', subset=None, inplace=False)
        ori_train, ori_test = train_test_split(sensor, test_size=0.5, random_state=seed, shuffle=False)
    elif data_set == enums.DataSetName.POWER_DEMAND.value:
        train_data, train_meta = arff.loadarff('ItalyPowerDemand_TRAIN.arff')
        test_data, test_meta = arff.loadarff('ItalyPowerDemand_TEST.arff')
        data, data_test = pd.DataFrame(train_data), pd.DataFrame(test_data)
        data['target'] = data['target'].map({b'1': 0, b'2': 1})
        data_test['target'] = data_test['target'].map({b'1': 0, b'2': 1})
        start_time = pd.Timestamp("2023-01-01 00:00:00")
        end_time = start_time + pd.DateOffset(minutes=len(data) - 1)
        time_index = pd.date_range(start=start_time, end=end_time, freq='1min')
        data.insert(0, 'time', time_index)
        test_start_time = pd.Timestamp("2023-01-01 01:17:00")
        test_end_time = test_start_time + pd.DateOffset(minutes=len(data_test) - 1)
        test_time_index = pd.date_range(start=test_start_time, end=test_end_time, freq='1min')
        data_test.insert(0, 'time', test_time_index)
        ori_train, ori_test = data.copy(), data_test.copy()
    else:
        raise RuntimeError(f'cannot find data_set: {data_set}')

    return ori_train, ori_test





# ============  OPT  ============

def my_opts(data_set, data_arrange, method_name, opt_name, tolerance, drop_rate=0.25, nums=1):

    if data_set == enums.DataSetName.OCCUPANCY.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = make_occupancy_df(file_path='occupancy_data.csv', data_arrange=data_arrange)

    elif data_set == enums.DataSetName.AIR_QUALITY.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = make_air_df(file_path='air_data.csv', data_arrange=data_arrange)

    elif data_set == enums.DataSetName.PUMP_SENSOR.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = make_sensor_df(file_path='sensor.csv', data_arrange=data_arrange)

    elif data_set == enums.DataSetName.POWER_DEMAND.value:
        X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val,  dicts = make_PowerDemand_df(train_file_path='ItalyPowerDemand_TRAIN.arff', test_file_path='ItalyPowerDemand_TEST.arff', data_arrange=data_arrange)

    else:
        raise RuntimeError(f'cannot find data_set: {data_set}')



    bounds = [(0.0, 1.0)] * (len(X_training.columns) + 1)

    if opt_name == enums.OptType.ORDERLY.value:
        d, w, v, n, decimal_delta, decimal_window_size = opt_utils.orderly_direct_opt(X_df=X_training, y_df=y_training,
                                                                                  method_name=method_name,
                                                                                  dicts=dicts, bounds=bounds, tolerance=tolerance,
                                                                                  data_arrange=data_arrange)


    elif opt_name == enums.OptType.BOOTSTRAP.value:
        d, w, v, n, decimal_delta, decimal_window_size = opt_utils.bootstrap_corr(drop_rate=drop_rate, nums=nums, X_df=X_training,
                                                                              y_df=y_training, method_name=method_name,
                                                                              dicts=dicts, bounds=bounds, data_arrange=data_arrange, tolerance=tolerance)
    else:
        raise RuntimeError(f'cannot find opt_name: {opt_name}')

    return d, w, v, n, decimal_delta, decimal_window_size


def main():

    fire.Fire(my_opts)


if __name__ ==  '__main__':
    main()