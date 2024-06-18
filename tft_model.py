import os
import numpy as np
import pickle
import constants
import model_utils
import opt
from sklearn.model_selection import train_test_split
from model_utils import *


# ============  Define relevant variables  ============
seed = constants.RAMDOM_SEED
np.random.seed(constants.RAMDOM_SEED)
dataset_name = enums.DataSetName.OCCUPANCY.value
data_arrange = enums.DataArrange.FIXED.value
task_type = enums.TaskType.CLASSIFICATION.value
mi_type = enums.CorrMethod.CLASSIF_MI.value
data_dicts = constants.OCCUPANCY_DICTS
tolerance = enums.ToleranceType.MINUTE.value



# ========= Import data ==========
X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = opt.make_model_df(data_set=dataset_name, data_arrange=data_arrange)
col_nums=len(X_training.columns)
ori_train, ori_test = opt.make_origin_df(data_set=dataset_name)

if dataset_name == enums.DataSetName.PUMP_SENSOR.value:
    ori_train_ioc = ori_train.iloc[69000:80000, :]
    ori_train = ori_train_ioc[:int(len(ori_train_ioc) * 0.5)]
    X_training, y_training = X_train, y_train




# ========= Import opt results ==========
o_cc_pkl = open(f'orderly_{data_arrange}_training_col{col_nums}_cc_test1.pkl', 'rb')
o_cc = pickle.load(o_cc_pkl)
o_mi_pkl = open(f'orderly_{data_arrange}_training_col{col_nums}_{mi_type}_test1.pkl', 'rb')
o_mi = pickle.load(o_mi_pkl)


b_cc_pkl1 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test1.pkl', 'rb')
b_cc_pkl2 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test2.pkl', 'rb')
b_cc_pkl3 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test3.pkl', 'rb')
b_cc_pkl4 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test4.pkl', 'rb')
b_cc_pkl5 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test5.pkl', 'rb')
b_cc1 = pickle.load(b_cc_pkl1)
b_cc2 = pickle.load(b_cc_pkl2)
b_cc3 = pickle.load(b_cc_pkl3)
b_cc4 = pickle.load(b_cc_pkl4)
b_cc5 = pickle.load(b_cc_pkl5)
b_cc_lst = [b_cc1, b_cc2, b_cc3, b_cc4, b_cc5]

b_mi_pkl1 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test1.pkl', 'rb')
b_mi_pkl2 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test2.pkl', 'rb')
b_mi_pkl3 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test3.pkl', 'rb')
b_mi_pkl4 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test4.pkl', 'rb')
b_mi_pkl5 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test5.pkl', 'rb')
b_mi1 = pickle.load(b_mi_pkl1)
b_mi2 = pickle.load(b_mi_pkl2)
b_mi3 = pickle.load(b_mi_pkl3)
b_mi4 = pickle.load(b_mi_pkl4)
b_mi5 = pickle.load(b_mi_pkl5)
b_mi_lst = [b_mi1, b_mi2, b_mi3, b_mi4, b_mi5]



# ============  Save dataframe as a pickle file  ============

cc_indicator = model_utils.single_tft_model(task_type=task_type, data_dicts=data_dicts,
                                         delta=o_cc['avg_delta'], window_size=o_cc['avg_window_size'],
                                         train_X_df=X_training, train_y_df=y_training, test_X_df=X_test, test_y_df=y_test, tolerance=tolerance)
mi_indicator = model_utils.single_tft_model(task_type=task_type, data_dicts=data_dicts,
                                         delta=o_mi['avg_delta'], window_size=o_mi['avg_window_size'],
                                         train_X_df=X_training, train_y_df=y_training, test_X_df=X_test, test_y_df=y_test, tolerance=tolerance)
no_alignment_indicator = model_utils.single_tft_model(task_type=task_type, data_dicts=data_dicts,
                                         delta=[0] * (len(X_training.columns) - 1), window_size=1,
                                         train_X_df=X_training, train_y_df=y_training, test_X_df=X_test, test_y_df=y_test, tolerance=tolerance)
real_time_delay_indicator = model_utils.single_tft_model(task_type=task_type, data_dicts=data_dicts, train_df=ori_train, test_df=ori_test, tolerance=tolerance)



result_dict = model_utils.make_orderly_tft_result_dict(cc_indicator=cc_indicator, mi_indicator=mi_indicator,
                                                       no_alignment_indicator=no_alignment_indicator,
                                                       real_time_delay_indicator=real_time_delay_indicator)
print(f'orderly_{data_arrange}_tft_result_dicts recording completed')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


times = 1
file = f'{dataset_name}_orderly_{data_arrange}_tft_result_test{times}.pkl'
if os.path.exists(file) == False:
    with open(file=file, mode="wb") as f:
        pickle.dump(result_dict, f, True)
else:
    while os.path.exists(file):
        times += 1
        new_file = f'{dataset_name}_orderly_{data_arrange}_tft_result_test{times}.pkl'
        if os.path.exists(new_file) == False:
            with open(file=new_file, mode="wb") as f:
                pickle.dump(result_dict, f, True)
        else:
            continue
        break




for td in range(len(b_cc_lst)):
    cc_max_df = model_utils.bootstrap_tft_model(task_type=task_type, data_dicts=data_dicts,
                                 bootstrap_td_lst=b_cc_lst[td], train_X_df=X_training, train_y_df=y_training,
                                 test_X_df=X_test, test_y_df=y_test, tolerance=tolerance)
    mi_max_df = model_utils.bootstrap_tft_model(task_type=task_type, data_dicts=data_dicts,
                                    bootstrap_td_lst=b_mi_lst[td], train_X_df=X_training, train_y_df=y_training,
                                    test_X_df=X_test, test_y_df=y_test, tolerance=tolerance)
    cc_max_df.to_pickle(f"{data_arrange}_{dataset_name}_tft_cc_df_{td}.pkl")
    mi_max_df.to_pickle(f"{data_arrange}_{dataset_name}_tft_mi_df_{td}.pkl")

