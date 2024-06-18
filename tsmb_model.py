import pickle
import constants
import model_utils
import enums
import os
import datetime
import opt


# ============  Define relevant variables  ============
seed = constants.RAMDOM_SEED
dataset_name = enums.DataSetName.POWER_DEMAND.value
data_arrange = enums.DataArrange.FIXED.value
task_type = enums.TaskType.CLASSIFICATION.value
mi_type = enums.CorrMethod.CLASSIF_MI.value
tolerance = enums.ToleranceType.MINUTE.value


# ========= Import data ==========
X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = opt.make_model_df(data_set=dataset_name, data_arrange=data_arrange)
col_nums=len(X_training.columns)
ori_train, ori_test = opt.make_origin_df(data_set=dataset_name)



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






# ============  Record results  ============

orderly_result_dicts = {}
cc_results = model_utils.orderly_model(task_type=task_type, val_or_test=enums.ModelDataType.TEST.value, delta=o_cc['avg_delta'],
                                       window_size=o_cc['avg_window_size'],
                                       X_training=X_training, y_training=y_training, X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val, tolerance=tolerance)

mi_results = model_utils.orderly_model(task_type=task_type, val_or_test=enums.ModelDataType.TEST.value, delta=o_mi['avg_delta'],
                                       window_size=o_mi['avg_window_size'],
                                       X_training=X_training, y_training=y_training, X_train=X_train,
                                       y_train=y_train,
                                       X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val, tolerance=tolerance)

no_alignment_results = model_utils.orderly_model(task_type=task_type, val_or_test=enums.ModelDataType.TEST.value,
                                                 delta=[0] * (len(X_training.columns) - 1), window_size=1,
                                                 X_training=X_training, y_training=y_training, X_train=X_train,
                                                 y_train=y_train,
                                                 X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val, tolerance=tolerance)

real_time_delay_results = model_utils.gdbt_classifier_model(train=ori_train, test=ori_test) if task_type == enums.TaskType.CLASSIFICATION.value else model_utils.gdbt_regressor_model(train=ori_train, test=ori_test)


orderly_result_dict = {'orderly_cc_indicator': cc_results[0], 'orderly_mi_indicator': mi_results[0],
                            'orderly_no_alignment_indicator': no_alignment_results[0],
                            'orderly_real_time_delay_indicator': real_time_delay_results[0]}
orderly_result_dicts[f'orderly_{enums.ModelDataType.TEST.value}_result_dict'] = orderly_result_dict

print(f'orderly_{data_arrange}_result_dicts recording completed')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


times = 1
file = f'{dataset_name}_orderly_{data_arrange}_result_test{times}.pkl'
if os.path.exists(file) == False:
    with open(file=file, mode="wb") as f:
        pickle.dump(orderly_result_dicts, f, True)
else:
    while os.path.exists(file):
        times += 1
        new_file = f'{dataset_name}_orderly_{data_arrange}_result_test{times}.pkl'
        if os.path.exists(new_file) == False:
            with open(file=new_file, mode="wb") as f:
                pickle.dump(orderly_result_dicts, f, True)
        else:
            continue
        break





all_cc_test_max_df, all_cc_val_max_df = model_utils.get_bootstrap_results(task_type=task_type, pkl_lst = b_cc_lst,
                                                               X_training=X_training, y_training=y_training,
                                                               X_train=X_train, y_train=y_train,
                                                               X_val=X_val, y_val=y_val,
                                                               X_test=X_test, y_test=y_test, dataset_name=dataset_name,
                                                               method_name=enums.CorrMethod.GCC.value, data_arrange=data_arrange, tolerance=tolerance)

all_mi_test_max_df, all_mi_val_max_df = model_utils.get_bootstrap_results(task_type=task_type, pkl_lst = b_mi_lst,
                                                               X_training=X_training, y_training=y_training,
                                                               X_train=X_train, y_train=y_train,
                                                               X_val=X_val, y_val=y_val,
                                                               X_test=X_test, y_test=y_test, dataset_name=dataset_name,
                                                               method_name=enums.CorrMethod.TDMI.value, data_arrange=data_arrange, tolerance=tolerance)
