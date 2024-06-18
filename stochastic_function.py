import numpy as np
import pandas as pd
import constants
from datetime import timedelta
from functools import reduce

seed = constants.RAMDOM_SEED


def get_random_idx_df(df):
    RandomIdxDf = df.copy()
    arr = np.array([0, 1, 2, 3, 4])
    probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    random_with_probs = np.random.choice(arr, size=len(df), p=probabilities)
    RandomIdxDf['idx'] = random_with_probs

    return RandomIdxDf



def get_air_quality_df_lsts(delta_lsts, df):
    delta_df_lst = []
    lable_df = df[['time', 'CO(GT)']]

    for delta_lst in delta_lsts:
        df0 = df[['time', 'PT08.S1(CO)']]
        df1 = df[['time', 'PT08.S2(NMHC)']]
        df2 = df[['time', 'PT08.S3(NOx)']]
        df3 = df[['time', 'PT08.S4(NO2)']]
        df4 = df[['time', 'PT08.S5(O3)']]
        df5 = df[['time', 'T']]
        df6 = df[['time', 'RH']]
        df7 = df[['time', 'AH']]
        df_lst = [df0, df1, df2, df3, df4, df5, df6, df7]

        for i in range(len(df.columns)-3):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=delta_lst[i])

        df_lst.append(lable_df)
        delta_df = reduce(lambda left, right: pd.merge(left, right, on=['time'], how='right'), df_lst)
        delta_df_lst.append(delta_df)

    return delta_df_lst



def get_occupancy_df_lsts(delta_lsts, df):
    delta_df_lst = []
    lable_df = df[['time', 'Occupancy']]

    for delta_lst in delta_lsts:
        df0 = df[['time', 'Temperature']]
        df1 = df[['time', 'Humidity']]
        df2 = df[['time', 'Light']]
        df3 = df[['time', 'CO2']]
        df4 = df[['time', 'HumidityRatio']]
        df_lst = [df0, df1, df2, df3, df4]

        for i in range(len(df.columns)-3):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=delta_lst[i])

        df_lst.append(lable_df)
        delta_df = reduce(lambda left, right: pd.merge(left, right, on=['time'], how='right'), df_lst)
        delta_df_lst.append(delta_df)

    return delta_df_lst



def get_pump_sensor_df_lsts(delta_lsts, df):
    delta_df_lst = []
    lable_df = df[['time', 'machine_status']]

    for delta_lst in delta_lsts:
        df0 = df[['time', 'sensor_04']]
        df1 = df[['time', 'sensor_06']]
        df2 = df[['time', 'sensor_07']]
        df3 = df[['time', 'sensor_08']]
        df4 = df[['time', 'sensor_09']]
        df5 = df[['time', 'sensor_10']]
        df_lst = [df0, df1, df2, df3, df4, df5]

        for i in range(len(df.columns)-3):
            df_lst[i].iloc[:, 0] = df_lst[i].iloc[:, 0] - timedelta(minutes=delta_lst[i])

        df_lst.append(lable_df)
        delta_df = reduce(lambda left, right: pd.merge(left, right, on=['time'], how='right'), df_lst)
        delta_df_lst.append(delta_df)

    return delta_df_lst



def get_power_demand_df_lsts(delta_lsts, data, data_test):
    data_delta_df_lst = []
    data_test_delta_df_lst = []

    feature_groups = [
        data.columns[1:9],
        data.columns[9:17],
        data.columns[17:25]
    ]

    for delta_lst in delta_lsts:

        data_copy = data.copy()
        data_test_copy = data_test.copy()

        time_deltas = delta_lst
        for group, time_delta in zip(feature_groups, time_deltas):
            for column in group:
                data_copy[column] = data_copy[column].shift(periods=-time_delta)
        data_delta_df_lst.append(data_copy)

        for group, time_delta in zip(feature_groups, time_deltas):
            for column in group:
                data_test_copy[column] = data_test_copy[column].shift(periods=-time_delta)
        data_test_delta_df_lst.append(data_test_copy)

    return data_delta_df_lst, data_test_delta_df_lst



def get_sorted_df(df, delta_df_lst):
    subdf_lsts = []

    for idx in range(constants.BOOTSTRAP_ITERATIONS):
        ori_df = df[df['idx'] == idx].iloc[:, [0]]
        sub_df = pd.merge(ori_df, delta_df_lst[idx], on=['time'], how='left')
        sub_df.dropna(inplace=True)
        subdf_lsts.append(sub_df)

    concat_df = pd.concat(subdf_lsts)
    sorted_df = concat_df.sort_values(by='time')
    return sorted_df

