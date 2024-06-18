import pickle
import constants
import model_utils
import opt
import enums

# ============  Define relevant variables  ============
seed = constants.RAMDOM_SEED
dataset_name = enums.DataSetName.OCCUPANCY.value
data_arrange = enums.DataArrange.FIXED.value
task_type = enums.TaskType.CLASSIFICATION.value
mi_type = enums.CorrMethod.TDMI.value
tolerance = enums.ToleranceType.MINUTE.value


# ========= Import data ==========
X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = opt.make_model_df(data_set=dataset_name, data_arrange=data_arrange)
col_nums=len(X_training.columns)


# ========= Import opt results ==========

p_pkl1 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test1.pkl', 'rb')
p_pkl2 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test2.pkl', 'rb')
p_pkl3 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test3.pkl', 'rb')
p_pkl4 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test4.pkl', 'rb')
p_pkl5 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test5.pkl', 'rb')
p1 = pickle.load(p_pkl1)
p2 = pickle.load(p_pkl2)
p3 = pickle.load(p_pkl3)
p4 = pickle.load(p_pkl4)
p5 = pickle.load(p_pkl5)

p_delta_cc1 = p1['cc_perturbed_delta']
p_delta_cc2 = p2['cc_perturbed_delta']
p_delta_cc3 = p3['cc_perturbed_delta']
p_delta_cc4 = p4['cc_perturbed_delta']
p_delta_cc5 = p5['cc_perturbed_delta']
p_cc_delta_lst = [p_delta_cc1, p_delta_cc2, p_delta_cc3, p_delta_cc4, p_delta_cc5]

p_window_size_cc1 = p1['cc_perturbed_window_size']
p_window_size_cc2 = p2['cc_perturbed_window_size']
p_window_size_cc3 = p3['cc_perturbed_window_size']
p_window_size_cc4 = p4['cc_perturbed_window_size']
p_window_size_cc5 = p5['cc_perturbed_window_size']
p_cc_window_size_lst = [p_window_size_cc1, p_window_size_cc2, p_window_size_cc3, p_window_size_cc4, p_window_size_cc5]

####################

p_delta_mi1 = p1['mi_perturbed_delta']
p_delta_mi2 = p2['mi_perturbed_delta']
p_delta_mi3 = p3['mi_perturbed_delta']
p_delta_mi4 = p4['mi_perturbed_delta']
p_delta_mi5 = p5['mi_perturbed_delta']
p_mi_delta_lst = [p_delta_mi1, p_delta_mi2, p_delta_mi3, p_delta_mi4, p_delta_mi5]

p_window_size_mi1 = p1['mi_perturbed_window_size']
p_window_size_mi2 = p2['mi_perturbed_window_size']
p_window_size_mi3 = p3['mi_perturbed_window_size']
p_window_size_mi4 = p4['mi_perturbed_window_size']
p_window_size_mi5 = p5['mi_perturbed_window_size']
p_mi_window_size_lst = [p_window_size_mi1, p_window_size_mi2, p_window_size_mi3, p_window_size_mi4, p_window_size_mi5]




# ============  Record results  ============

all_cc_test_max_df, all_cc_val_max_df = model_utils.get_perturbed_bootstrap_results(task_type=task_type, delta_pkl_lst= p_cc_delta_lst , window_size_pkl_lst=p_cc_window_size_lst,
                                                               X_training=X_training, y_training=y_training,
                                                               X_train=X_train, y_train=y_train,
                                                               X_val=X_val, y_val=y_val,
                                                               X_test=X_test, y_test=y_test, data_arrange=data_arrange, method_name=enums.CorrMethod.GCC.value, dataset_name=dataset_name, tolerance=tolerance)

all_mi_test_max_df, all_mi_val_max_df = model_utils.get_perturbed_bootstrap_results(task_type=task_type, delta_pkl_lst= p_mi_delta_lst , window_size_pkl_lst=p_mi_window_size_lst,
                                                               X_training=X_training, y_training=y_training,
                                                               X_train=X_train, y_train=y_train,
                                                               X_val=X_val, y_val=y_val,
                                                               X_test=X_test, y_test=y_test, data_arrange=data_arrange, method_name=enums.CorrMethod.TDMI.value, dataset_name=dataset_name, tolerance=tolerance)
