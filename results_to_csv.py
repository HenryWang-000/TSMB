import numpy as np
import pandas as pd
import pickle
import generate_results_function
import opt
import constants
import enums


# ============  Define relevant variables  ============
seed = constants.RAMDOM_SEED
np.random.seed(constants.RAMDOM_SEED)
dataset_name = enums.DataSetName.AIR_QUALITY.value
data_arrange = enums.DataArrange.STOCHASTIC.value
task_type = enums.TaskType.REGRESSION.value
mi_type = enums.CorrMethod.TDMI.value
td_range = constants.AIR_QUALITY_TD_RANGE
data_dicts = constants.AIR_QUALITY_DICTS
tolerance = enums.ToleranceType.TWELVEHOURS.value


# ========= Import data ==========
X_training, y_training, X_train, y_train, X_test, y_test, X_val, y_val, dicts = opt.make_model_df(data_set=dataset_name, data_arrange=data_arrange)
col_nums=len(X_training.columns)
ori_train, ori_test = opt.make_origin_df(data_set=dataset_name)



# ========= Import Model Results ==========

b_cc_pkl1 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test1.pkl', 'rb')
b_cc_pkl2 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test2.pkl', 'rb')
b_cc_pkl3 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test3.pkl', 'rb')
b_cc_pkl4 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test4.pkl', 'rb')
b_cc_pkl5 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_cc_nums100_test5.pkl', 'rb')
b_cc1, b_cc2, b_cc3, b_cc4, b_cc5 = pickle.load(b_cc_pkl1), pickle.load(b_cc_pkl2), pickle.load(b_cc_pkl3), pickle.load(b_cc_pkl4), pickle.load(b_cc_pkl5)
b_cc_lst = [b_cc1, b_cc2, b_cc3, b_cc4, b_cc5]
b_cc_pkl1.close()
b_cc_pkl2.close()
b_cc_pkl3.close()
b_cc_pkl4.close()
b_cc_pkl5.close()

b_mi_pkl1 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test1.pkl', 'rb')
b_mi_pkl2 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test2.pkl', 'rb')
b_mi_pkl3 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test3.pkl', 'rb')
b_mi_pkl4 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test4.pkl', 'rb')
b_mi_pkl5 = open(f'bootstrap_{data_arrange}_training_col{col_nums}_{mi_type}_nums100_test5.pkl', 'rb')
b_mi1, b_mi2, b_mi3, b_mi4, b_mi5 = pickle.load(b_mi_pkl1), pickle.load(b_mi_pkl2), pickle.load(b_mi_pkl3), pickle.load(b_mi_pkl4), pickle.load(b_mi_pkl5)
b_mi_lst = [b_mi1, b_mi2, b_mi3, b_mi4, b_mi5]
b_mi_pkl1.close()
b_mi_pkl2.close()
b_mi_pkl3.close()
b_mi_pkl4.close()
b_mi_pkl5.close()


orderly_pkl = open(f'{dataset_name}_orderly_{data_arrange}_result_test1.pkl', 'rb')
orderly = pickle.load(orderly_pkl)
orderly_pkl.close()

perturbed_pkl1 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test1.pkl', 'rb')
perturbed_pkl2 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test2.pkl', 'rb')
perturbed_pkl3 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test3.pkl', 'rb')
perturbed_pkl4 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test4.pkl', 'rb')
perturbed_pkl5 = open(f'{dataset_name}_{data_arrange}_perturbed_result_test5.pkl', 'rb')
perturbed1 = pickle.load(perturbed_pkl1)
perturbed2 = pickle.load(perturbed_pkl2)
perturbed3 = pickle.load(perturbed_pkl3)
perturbed4 = pickle.load(perturbed_pkl4)
perturbed5 = pickle.load(perturbed_pkl5)
perturbed_pkl1.close()
perturbed_pkl2.close()
perturbed_pkl3.close()
perturbed_pkl4.close()
perturbed_pkl5.close()

cc_test_0_pkl = open(f'{data_arrange}_{dataset_name}_cc_test_max_df_0.pkl', 'rb')
cc_val_0_pkl = open(f'{data_arrange}_{dataset_name}_cc_val_max_df_0.pkl', 'rb')
cc_test_1_pkl = open(f'{data_arrange}_{dataset_name}_cc_test_max_df_1.pkl', 'rb')
cc_val_1_pkl = open(f'{data_arrange}_{dataset_name}_cc_val_max_df_1.pkl', 'rb')
cc_test_2_pkl = open(f'{data_arrange}_{dataset_name}_cc_test_max_df_2.pkl', 'rb')
cc_val_2_pkl = open(f'{data_arrange}_{dataset_name}_cc_val_max_df_2.pkl', 'rb')
cc_test_3_pkl = open(f'{data_arrange}_{dataset_name}_cc_test_max_df_3.pkl', 'rb')
cc_val_3_pkl = open(f'{data_arrange}_{dataset_name}_cc_val_max_df_3.pkl', 'rb')
cc_test_4_pkl = open(f'{data_arrange}_{dataset_name}_cc_test_max_df_4.pkl', 'rb')
cc_val_4_pkl = open(f'{data_arrange}_{dataset_name}_cc_val_max_df_4.pkl', 'rb')

cc_test_0 = pickle.load(cc_test_0_pkl)
cc_val_0 = pickle.load(cc_val_0_pkl)
cc_val_0 = cc_val_0.sort_values(cc_val_0.columns[0])
cc_test_0 = cc_test_0.sort_values(cc_test_0.columns[0])

cc_test_1 = pickle.load(cc_test_1_pkl)
cc_val_1 = pickle.load(cc_val_1_pkl)
cc_val_1 = cc_val_1.sort_values(cc_val_1.columns[0])
cc_test_1 = cc_test_1.sort_values(cc_test_1.columns[0])

cc_test_2 = pickle.load(cc_test_2_pkl)
cc_val_2 = pickle.load(cc_val_2_pkl)
cc_val_2 = cc_val_2.sort_values(cc_val_2.columns[0])
cc_test_2 = cc_test_2.sort_values(cc_test_2.columns[0])

cc_test_3 = pickle.load(cc_test_3_pkl)
cc_val_3 = pickle.load(cc_val_3_pkl)
cc_val_3 = cc_val_3.sort_values(cc_val_3.columns[0])
cc_test_3 = cc_test_3.sort_values(cc_test_3.columns[0])

cc_test_4 = pickle.load(cc_test_4_pkl)
cc_val_4 = pickle.load(cc_val_4_pkl)
cc_val_4 = cc_val_4.sort_values(cc_val_4.columns[0])
cc_test_4 = cc_test_4.sort_values(cc_test_4.columns[0])

cc_test_0_pkl.close()
cc_val_0_pkl.close()
cc_test_1_pkl.close()
cc_val_1_pkl.close()
cc_test_2_pkl.close()
cc_val_2_pkl.close()
cc_test_3_pkl.close()
cc_val_3_pkl.close()
cc_test_4_pkl.close()
cc_val_4_pkl.close()

cc_val_df_lst = [cc_val_0, cc_val_1, cc_val_2, cc_val_3, cc_val_4]
cc_test_df_lst = [cc_test_0, cc_test_1, cc_test_2, cc_test_3, cc_test_4]



mi_test_0_pkl = open(f'{data_arrange}_{dataset_name}_mi_test_max_df_0.pkl', 'rb')
mi_val_0_pkl = open(f'{data_arrange}_{dataset_name}_mi_val_max_df_0.pkl', 'rb')
mi_test_1_pkl = open(f'{data_arrange}_{dataset_name}_mi_test_max_df_1.pkl', 'rb')
mi_val_1_pkl = open(f'{data_arrange}_{dataset_name}_mi_val_max_df_1.pkl', 'rb')
mi_test_2_pkl = open(f'{data_arrange}_{dataset_name}_mi_test_max_df_2.pkl', 'rb')
mi_val_2_pkl = open(f'{data_arrange}_{dataset_name}_mi_val_max_df_2.pkl', 'rb')
mi_test_3_pkl = open(f'{data_arrange}_{dataset_name}_mi_test_max_df_3.pkl', 'rb')
mi_val_3_pkl = open(f'{data_arrange}_{dataset_name}_mi_val_max_df_3.pkl', 'rb')
mi_test_4_pkl = open(f'{data_arrange}_{dataset_name}_mi_test_max_df_4.pkl', 'rb')
mi_val_4_pkl = open(f'{data_arrange}_{dataset_name}_mi_val_max_df_4.pkl', 'rb')

mi_test_0 = pickle.load(mi_test_0_pkl)
mi_val_0 = pickle.load(mi_val_0_pkl)
mi_val_0 = mi_val_0.sort_values(mi_val_0.columns[0])
mi_test_0 = mi_test_0.sort_values(mi_test_0.columns[0])

mi_test_1 = pickle.load(mi_test_1_pkl)
mi_val_1 = pickle.load(mi_val_1_pkl)
mi_val_1 = mi_val_1.sort_values(mi_val_1.columns[0])
mi_test_1 = mi_test_1.sort_values(mi_test_1.columns[0])

mi_test_2 = pickle.load(mi_test_2_pkl)
mi_val_2 = pickle.load(mi_val_2_pkl)
mi_val_2 = mi_val_2.sort_values(mi_val_2.columns[0])
mi_test_2 = mi_test_2.sort_values(mi_test_2.columns[0])

mi_test_3 = pickle.load(mi_test_3_pkl)
mi_val_3 = pickle.load(mi_val_3_pkl)
mi_val_3 = mi_val_3.sort_values(mi_val_3.columns[0])
mi_test_3 = mi_test_3.sort_values(mi_test_3.columns[0])

mi_test_4 = pickle.load(mi_test_4_pkl)
mi_val_4 = pickle.load(mi_val_4_pkl)
mi_val_4 = mi_val_4.sort_values(mi_val_4.columns[0])
mi_test_4 = mi_test_4.sort_values(mi_test_4.columns[0])

mi_test_0_pkl.close()
mi_val_0_pkl.close()
mi_test_1_pkl.close()
mi_val_1_pkl.close()
mi_test_2_pkl.close()
mi_val_2_pkl.close()
mi_test_3_pkl.close()
mi_val_3_pkl.close()
mi_test_4_pkl.close()
mi_val_4_pkl.close()

mi_val_df_lst = [mi_val_0, mi_val_1, mi_val_2, mi_val_3, mi_val_4]
mi_test_df_lst = [mi_test_0, mi_test_1, mi_test_2, mi_test_3, mi_test_4]

o_cc_pkl = open(f'orderly_{data_arrange}_training_col{col_nums}_cc_test1.pkl', 'rb')
o_cc = pickle.load(o_cc_pkl)
o_cc_pkl.close()
o_mi_pkl = open(f'orderly_{data_arrange}_training_col{col_nums}_{mi_type}_test1.pkl', 'rb')
o_mi = pickle.load(o_mi_pkl)
o_mi_pkl.close()


p_cc_test_0_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_test_max_df_0.pkl', 'rb')
p_cc_val_0_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_val_max_df_0.pkl', 'rb')
p_cc_test_1_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_test_max_df_1.pkl', 'rb')
p_cc_val_1_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_val_max_df_1.pkl', 'rb')
p_cc_test_2_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_test_max_df_2.pkl', 'rb')
p_cc_val_2_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_val_max_df_2.pkl', 'rb')
p_cc_test_3_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_test_max_df_3.pkl', 'rb')
p_cc_val_3_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_val_max_df_3.pkl', 'rb')
p_cc_test_4_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_test_max_df_4.pkl', 'rb')
p_cc_val_4_pkl = open(f'p_{data_arrange}_{dataset_name}_cc_val_max_df_4.pkl', 'rb')

p_cc_test_0 = pickle.load(p_cc_test_0_pkl)
p_cc_val_0 = pickle.load(p_cc_val_0_pkl)
p_cc_val_0 = p_cc_val_0.sort_values(p_cc_val_0.columns[0])
p_cc_test_0 = p_cc_test_0.sort_values(p_cc_test_0.columns[0])

p_cc_test_1 = pickle.load(p_cc_test_1_pkl)
p_cc_val_1 = pickle.load(p_cc_val_1_pkl)
p_cc_val_1 = p_cc_val_1.sort_values(p_cc_val_1.columns[0])
p_cc_test_1 = p_cc_test_1.sort_values(p_cc_test_1.columns[0])

p_cc_test_2 = pickle.load(p_cc_test_2_pkl)
p_cc_val_2 = pickle.load(p_cc_val_2_pkl)
p_cc_val_2 = p_cc_val_2.sort_values(p_cc_val_2.columns[0])
p_cc_test_2 = p_cc_test_2.sort_values(p_cc_test_2.columns[0])

p_cc_test_3 = pickle.load(p_cc_test_3_pkl)
p_cc_val_3 = pickle.load(p_cc_val_3_pkl)
p_cc_val_3 = p_cc_val_3.sort_values(p_cc_val_3.columns[0])
p_cc_test_3 = p_cc_test_3.sort_values(p_cc_test_3.columns[0])

p_cc_test_4 = pickle.load(p_cc_test_4_pkl)
p_cc_val_4 = pickle.load(p_cc_val_4_pkl)
p_cc_val_4 = p_cc_val_4.sort_values(p_cc_val_4.columns[0])
p_cc_test_4 = p_cc_test_4.sort_values(p_cc_test_4.columns[0])

p_cc_test_0_pkl.close()
p_cc_val_0_pkl.close()
p_cc_test_1_pkl.close()
p_cc_val_1_pkl.close()
p_cc_test_2_pkl.close()
p_cc_val_2_pkl.close()
p_cc_test_3_pkl.close()
p_cc_val_3_pkl.close()
p_cc_test_4_pkl.close()
p_cc_val_4_pkl.close()

p_cc_val_df_lst = [p_cc_val_0, p_cc_val_1, p_cc_val_2, p_cc_val_3, p_cc_val_4]
p_cc_test_df_lst = [p_cc_test_0, p_cc_test_1, p_cc_test_2, p_cc_test_3, p_cc_test_4]


p_mi_test_0_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_test_max_df_0.pkl', 'rb')
p_mi_val_0_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_val_max_df_0.pkl', 'rb')
p_mi_test_1_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_test_max_df_1.pkl', 'rb')
p_mi_val_1_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_val_max_df_1.pkl', 'rb')
p_mi_test_2_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_test_max_df_2.pkl', 'rb')
p_mi_val_2_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_val_max_df_2.pkl', 'rb')
p_mi_test_3_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_test_max_df_3.pkl', 'rb')
p_mi_val_3_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_val_max_df_3.pkl', 'rb')
p_mi_test_4_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_test_max_df_4.pkl', 'rb')
p_mi_val_4_pkl = open(f'p_{data_arrange}_{dataset_name}_mi_val_max_df_4.pkl', 'rb')

p_mi_test_0 = pickle.load(p_mi_test_0_pkl)
p_mi_val_0 = pickle.load(p_mi_val_0_pkl)
p_mi_val_0 = p_mi_val_0.sort_values(p_mi_val_0.columns[0])
p_mi_test_0 = p_mi_test_0.sort_values(p_mi_test_0.columns[0])

p_mi_test_1 = pickle.load(p_mi_test_1_pkl)
p_mi_val_1 = pickle.load(p_mi_val_1_pkl)
p_mi_val_1 = p_mi_val_1.sort_values(p_mi_val_1.columns[0])
p_mi_test_1 = p_mi_test_1.sort_values(p_mi_test_1.columns[0])

p_mi_test_2 = pickle.load(p_mi_test_2_pkl)
p_mi_val_2 = pickle.load(p_mi_val_2_pkl)
p_mi_val_2 = p_mi_val_2.sort_values(p_mi_val_2.columns[0])
p_mi_test_2 = p_mi_test_2.sort_values(p_mi_test_2.columns[0])

p_mi_test_3 = pickle.load(p_mi_test_3_pkl)
p_mi_val_3 = pickle.load(p_mi_val_3_pkl)
p_mi_val_3 = p_mi_val_3.sort_values(p_mi_val_3.columns[0])
p_mi_test_3 = p_mi_test_3.sort_values(p_mi_test_3.columns[0])

p_mi_test_4 = pickle.load(p_mi_test_4_pkl)
p_mi_val_4 = pickle.load(p_mi_val_4_pkl)
p_mi_val_4 = p_mi_val_4.sort_values(p_mi_val_4.columns[0])
p_mi_test_4 = p_mi_test_4.sort_values(p_mi_test_4.columns[0])

p_mi_test_0_pkl.close()
p_mi_val_0_pkl.close()
p_mi_test_1_pkl.close()
p_mi_val_1_pkl.close()
p_mi_test_2_pkl.close()
p_mi_val_2_pkl.close()
p_mi_test_3_pkl.close()
p_mi_val_3_pkl.close()
p_mi_test_4_pkl.close()
p_mi_val_4_pkl.close()
p_mi_val_df_lst = [p_mi_val_0, p_mi_val_1, p_mi_val_2, p_mi_val_3, p_mi_val_4]
p_mi_test_df_lst = [p_mi_test_0, p_mi_test_1, p_mi_test_2, p_mi_test_3, p_mi_test_4]

## Import TFT Model Results

tft_orderly_pkl = open(f'{dataset_name}_orderly_{data_arrange}_tft_result_test1.pkl', 'rb')
tft_orderly = pickle.load(tft_orderly_pkl)
tft_orderly_pkl.close()

tft_cc_test_0_pkl = open(f'{data_arrange}_{dataset_name}_tft_cc_df_0.pkl', 'rb')
tft_cc_test_1_pkl = open(f'{data_arrange}_{dataset_name}_tft_cc_df_1.pkl', 'rb')
tft_cc_test_2_pkl = open(f'{data_arrange}_{dataset_name}_tft_cc_df_2.pkl', 'rb')
tft_cc_test_3_pkl = open(f'{data_arrange}_{dataset_name}_tft_cc_df_3.pkl', 'rb')
tft_cc_test_4_pkl = open(f'{data_arrange}_{dataset_name}_tft_cc_df_4.pkl', 'rb')
tft_cc_test_0 = pickle.load(tft_cc_test_0_pkl)
tft_cc_test_0 = tft_cc_test_0.sort_values(tft_cc_test_0.columns[0])
tft_cc_test_1 = pickle.load(tft_cc_test_1_pkl)
tft_cc_test_1 = tft_cc_test_1.sort_values(tft_cc_test_1.columns[0])
tft_cc_test_2 = pickle.load(tft_cc_test_2_pkl)
tft_cc_test_2 = tft_cc_test_2.sort_values(tft_cc_test_2.columns[0])
tft_cc_test_3 = pickle.load(tft_cc_test_3_pkl)
tft_cc_test_3 = tft_cc_test_3.sort_values(tft_cc_test_3.columns[0])
tft_cc_test_4 = pickle.load(tft_cc_test_4_pkl)
tft_cc_test_4 = tft_cc_test_4.sort_values(tft_cc_test_4.columns[0])
tft_cc_test_0_pkl.close()
tft_cc_test_1_pkl.close()
tft_cc_test_2_pkl.close()
tft_cc_test_3_pkl.close()
tft_cc_test_4_pkl.close()
tft_cc_test_df_lst = [tft_cc_test_0, tft_cc_test_1, tft_cc_test_2, tft_cc_test_3, tft_cc_test_4]

tft_mi_test_0_pkl = open(f'{data_arrange}_{dataset_name}_tft_mi_df_0.pkl', 'rb')
tft_mi_test_1_pkl = open(f'{data_arrange}_{dataset_name}_tft_mi_df_1.pkl', 'rb')
tft_mi_test_2_pkl = open(f'{data_arrange}_{dataset_name}_tft_mi_df_2.pkl', 'rb')
tft_mi_test_3_pkl = open(f'{data_arrange}_{dataset_name}_tft_mi_df_3.pkl', 'rb')
tft_mi_test_4_pkl = open(f'{data_arrange}_{dataset_name}_tft_mi_df_4.pkl', 'rb')
tft_mi_test_0 = pickle.load(tft_mi_test_0_pkl)
tft_mi_test_0 = tft_mi_test_0.sort_values(tft_mi_test_0.columns[0])
tft_mi_test_1 = pickle.load(tft_mi_test_1_pkl)
tft_mi_test_1 = tft_mi_test_1.sort_values(tft_mi_test_1.columns[0])
tft_mi_test_2 = pickle.load(tft_mi_test_2_pkl)
tft_mi_test_2 = tft_mi_test_2.sort_values(tft_mi_test_2.columns[0])
tft_mi_test_3 = pickle.load(tft_mi_test_3_pkl)
tft_mi_test_3 = tft_mi_test_3.sort_values(tft_mi_test_3.columns[0])
tft_mi_test_4 = pickle.load(tft_mi_test_4_pkl)
tft_mi_test_4 = tft_mi_test_4.sort_values(tft_mi_test_4.columns[0])
tft_mi_test_0_pkl.close()
tft_mi_test_1_pkl.close()
tft_mi_test_2_pkl.close()
tft_mi_test_3_pkl.close()
tft_mi_test_4_pkl.close()
tft_mi_test_df_lst = [tft_mi_test_0, tft_mi_test_1, tft_mi_test_2, tft_mi_test_3, tft_mi_test_4]


# ============  Generate Results  ============
## ======  confidence  ======
y_true_cc = generate_results_function.make_predict_df(o_cc['avg_delta'], o_cc['avg_window_size'], X_training, y_training, X_test,
                                                   y_test, method_name=enums.CorrMethod.GCC.value, task_name=task_type, tolerance=tolerance)
y_true_mi = generate_results_function.make_predict_df(o_mi['avg_delta'], o_mi['avg_window_size'], X_training, y_training, X_test,
                                                   y_test, method_name=enums.CorrMethod.TDMI.value, task_name=task_type, tolerance=tolerance)
confidence_95_cc, confidence_90_cc, confidence_80_cc = generate_results_function.cofidence_func(cc_test_df_lst, y_true_cc, task_name=task_type)
confidence_95_mi, confidence_90_mi, confidence_80_mi = generate_results_function.cofidence_func(mi_test_df_lst, y_true_mi, task_name=task_type)
confidence_df = generate_results_function.make_confidence_df(dataset_name=dataset_name, cc_95=confidence_95_cc, cc_90=confidence_90_cc, cc_80=confidence_80_cc, mi_95=confidence_95_mi, mi_90=confidence_90_mi, mi_80=confidence_80_mi)

confidence_df.to_csv(f'{dataset_name}_{data_arrange}_tsmb_confidence.csv', index=False)
print("condifence completed")


## ======  distribution  ======
dis_cc = generate_results_function.distribution_func(dataset_name=dataset_name, b_lst=b_cc_lst, td_range =td_range, data_dicts=data_dicts, corr_name=enums.CorrMethod.TSMB_GCC.value, data_arrange=data_arrange)
dis_mi = generate_results_function.distribution_func(dataset_name=dataset_name, b_lst=b_mi_lst, td_range =td_range, data_dicts=data_dicts, corr_name=enums.CorrMethod.TSMB_TDMI.value, data_arrange=data_arrange)
dis_df = pd.concat([dis_mi, dis_cc], axis=0)

dis_df.to_csv(f'{dataset_name}_{data_arrange}_dis_tsmb.csv', index=False)
print("distribution completed")


## ======  top  ======
cc_tsmb_top1 = generate_results_function.tsmb_method(val_df_lst=cc_val_df_lst, test_df_lst=cc_test_df_lst, n=1, y_test=y_test, task_name=task_type)
cc_tsmb_top5 = generate_results_function.tsmb_method(val_df_lst=cc_val_df_lst, test_df_lst=cc_test_df_lst, n=5, y_test=y_test, task_name=task_type)
cc_tsmb_top10 = generate_results_function.tsmb_method(val_df_lst=cc_val_df_lst, test_df_lst=cc_test_df_lst, n=10, y_test=y_test, task_name=task_type)
cc_tsmb_top20 = generate_results_function.tsmb_method(val_df_lst=cc_val_df_lst, test_df_lst=cc_test_df_lst, n=20, y_test=y_test, task_name=task_type)
cc_tsmb_top50 = generate_results_function.tsmb_method(val_df_lst=cc_val_df_lst, test_df_lst=cc_test_df_lst, n=50, y_test=y_test, task_name=task_type)
cc_tsmb_top100 = generate_results_function.tsmb_method(val_df_lst=cc_val_df_lst, test_df_lst=cc_test_df_lst, n=100, y_test=y_test, task_name=task_type)
cc_tsmb_top_lst = [cc_tsmb_top1, cc_tsmb_top5, cc_tsmb_top10, cc_tsmb_top20, cc_tsmb_top50, cc_tsmb_top100]
mi_tsmb_top1 = generate_results_function.tsmb_method(val_df_lst=mi_val_df_lst, test_df_lst=mi_test_df_lst, n=1, y_test=y_test, task_name=task_type)
mi_tsmb_top5 = generate_results_function.tsmb_method(val_df_lst=mi_val_df_lst, test_df_lst=mi_test_df_lst, n=5, y_test=y_test, task_name=task_type)
mi_tsmb_top10 = generate_results_function.tsmb_method(val_df_lst=mi_val_df_lst, test_df_lst=mi_test_df_lst, n=10, y_test=y_test, task_name=task_type)
mi_tsmb_top20 = generate_results_function.tsmb_method(val_df_lst=mi_val_df_lst, test_df_lst=mi_test_df_lst, n=20, y_test=y_test, task_name=task_type)
mi_tsmb_top50 = generate_results_function.tsmb_method(val_df_lst=mi_val_df_lst, test_df_lst=mi_test_df_lst, n=50, y_test=y_test, task_name=task_type)
mi_tsmb_top100 = generate_results_function.tsmb_method(val_df_lst=mi_val_df_lst, test_df_lst=mi_test_df_lst, n=100, y_test=y_test, task_name=task_type)
mi_tsmb_top_lst = [mi_tsmb_top1, mi_tsmb_top5, mi_tsmb_top10, mi_tsmb_top20, mi_tsmb_top50, mi_tsmb_top100]


cc_perturbed_top1 = generate_results_function.tsmb_method(val_df_lst=p_cc_val_df_lst, test_df_lst=p_cc_test_df_lst, n=1, y_test=y_test, task_name=task_type)
cc_perturbed_top5 = generate_results_function.tsmb_method(val_df_lst=p_cc_val_df_lst, test_df_lst=p_cc_test_df_lst, n=5, y_test=y_test, task_name=task_type)
cc_perturbed_top10 = generate_results_function.tsmb_method(val_df_lst=p_cc_val_df_lst, test_df_lst=p_cc_test_df_lst, n=10, y_test=y_test, task_name=task_type)
cc_perturbed_top20 = generate_results_function.tsmb_method(val_df_lst=p_cc_val_df_lst, test_df_lst=p_cc_test_df_lst, n=20, y_test=y_test, task_name=task_type)
cc_perturbed_top50 = generate_results_function.tsmb_method(val_df_lst=p_cc_val_df_lst, test_df_lst=p_cc_test_df_lst, n=50, y_test=y_test, task_name=task_type)
cc_perturbed_top100 = generate_results_function.tsmb_method(val_df_lst=p_cc_val_df_lst, test_df_lst=p_cc_test_df_lst, n=100, y_test=y_test, task_name=task_type)
cc_perturbed_top_lst = [cc_perturbed_top1, cc_perturbed_top5, cc_perturbed_top10, cc_perturbed_top20, cc_perturbed_top50, cc_perturbed_top100]
mi_perturbed_top1 = generate_results_function.tsmb_method(val_df_lst=p_mi_val_df_lst, test_df_lst=p_mi_test_df_lst, n=1, y_test=y_test, task_name=task_type)
mi_perturbed_top5 = generate_results_function.tsmb_method(val_df_lst=p_mi_val_df_lst, test_df_lst=p_mi_test_df_lst, n=5, y_test=y_test, task_name=task_type)
mi_perturbed_top10 = generate_results_function.tsmb_method(val_df_lst=p_mi_val_df_lst, test_df_lst=p_mi_test_df_lst, n=10, y_test=y_test, task_name=task_type)
mi_perturbed_top20 = generate_results_function.tsmb_method(val_df_lst=p_mi_val_df_lst, test_df_lst=p_mi_test_df_lst, n=20, y_test=y_test, task_name=task_type)
mi_perturbed_top50 = generate_results_function.tsmb_method(val_df_lst=p_mi_val_df_lst, test_df_lst=p_mi_test_df_lst, n=50, y_test=y_test, task_name=task_type)
mi_perturbed_top100 = generate_results_function.tsmb_method(val_df_lst=p_mi_val_df_lst, test_df_lst=p_mi_test_df_lst, n=100, y_test=y_test, task_name=task_type)
mi_perturbed_top_lst = [mi_perturbed_top1, mi_perturbed_top5, mi_perturbed_top10, mi_perturbed_top20, mi_perturbed_top50, mi_perturbed_top100]


cc_tdb_top1 = generate_results_function.tdb_method(val_df_lst=cc_val_df_lst, b_lst=b_cc_lst, n=1, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_top5 = generate_results_function.tdb_method(val_df_lst=cc_val_df_lst, b_lst=b_cc_lst, n=5, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_top10 = generate_results_function.tdb_method(val_df_lst=cc_val_df_lst, b_lst=b_cc_lst, n=10, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_top20 = generate_results_function.tdb_method(val_df_lst=cc_val_df_lst, b_lst=b_cc_lst, n=20, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_top50 = generate_results_function.tdb_method(val_df_lst=cc_val_df_lst, b_lst=b_cc_lst, n=50, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_top100 = generate_results_function.tdb_method(val_df_lst=cc_val_df_lst, b_lst=b_cc_lst, n=100, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_top_lst = [cc_tdb_top1, cc_tdb_top5, cc_tdb_top10, cc_tdb_top20, cc_tdb_top50, cc_tdb_top100]
mi_tdb_top1 = generate_results_function.tdb_method(val_df_lst=mi_val_df_lst, b_lst=b_mi_lst, n=1, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_top5 = generate_results_function.tdb_method(val_df_lst=mi_val_df_lst, b_lst=b_mi_lst, n=5, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_top10 = generate_results_function.tdb_method(val_df_lst=mi_val_df_lst, b_lst=b_mi_lst, n=10, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_top20 = generate_results_function.tdb_method(val_df_lst=mi_val_df_lst, b_lst=b_mi_lst, n=20, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_top50 = generate_results_function.tdb_method(val_df_lst=mi_val_df_lst, b_lst=b_mi_lst, n=50, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_top100 = generate_results_function.tdb_method(val_df_lst=mi_val_df_lst, b_lst=b_mi_lst, n=100, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_top_lst = [mi_tdb_top1, mi_tdb_top5, mi_tdb_top10, mi_tdb_top20, mi_tdb_top50, mi_tdb_top100]


tsmb_top =  generate_results_function.make_top_df(all_cc_top_lst=cc_tsmb_top_lst, all_mi_top_lst=mi_tsmb_top_lst, method_type=enums.MethodType.TSMB.value, dataset_name=dataset_name)
# tsmb_top['test_values'] = tsmb_top['test_values'] - orderly['orderly_test_result_dict']['orderly_no_alignment_indicator']
print("tsmb top completed")
perturbed_top =  generate_results_function.make_top_df(all_cc_top_lst=cc_perturbed_top_lst, all_mi_top_lst=mi_perturbed_top_lst, method_type=enums.MethodType.PERTURBED_MODEL.value, dataset_name=dataset_name)
# perturbed_top['test_values'] = perturbed_top['test_values'] - orderly['orderly_test_result_dict']['orderly_no_alignment_indicator']
print("perturbed top completed")
tdb_top = generate_results_function.make_top_df(all_cc_top_lst=cc_tdb_top_lst, all_mi_top_lst=mi_tdb_top_lst, method_type=enums.MethodType.TDB.value, dataset_name=dataset_name)
# tdb_top['test_values'] = tdb_top['test_values'] - orderly['orderly_test_result_dict']['orderly_no_alignment_indicator']
print("tdb top completed")

tsmb_top.to_csv(f'{dataset_name}_{data_arrange}_tsmb_top.csv', index=False)
perturbed_top.to_csv(f'{dataset_name}_{data_arrange}_perturbed_top.csv', index=False)
tdb_top.to_csv(f'{dataset_name}_{data_arrange}_tdb_top.csv', index=False)



## ======  select B  ======
cc_tsmb_B1 = generate_results_function.tsmb_select_b_func(test_df_lst=cc_test_df_lst, B=1, y_test=y_test, task_name=task_type)
cc_tsmb_B5 = generate_results_function.tsmb_select_b_func(test_df_lst=cc_test_df_lst, B=5, y_test=y_test, task_name=task_type)
cc_tsmb_B10 = generate_results_function.tsmb_select_b_func(test_df_lst=cc_test_df_lst, B=10, y_test=y_test, task_name=task_type)
cc_tsmb_B20 = generate_results_function.tsmb_select_b_func(test_df_lst=cc_test_df_lst, B=20, y_test=y_test, task_name=task_type)
cc_tsmb_B50 = generate_results_function.tsmb_select_b_func(test_df_lst=cc_test_df_lst, B=50, y_test=y_test, task_name=task_type)
cc_tsmb_B100 = generate_results_function.tsmb_select_b_func(test_df_lst=cc_test_df_lst, B=100, y_test=y_test, task_name=task_type)
cc_tsmb_select_b_lst = [cc_tsmb_B1, cc_tsmb_B5, cc_tsmb_B10, cc_tsmb_B20, cc_tsmb_B50, cc_tsmb_B100]
mi_tsmb_B1 = generate_results_function.tsmb_select_b_func(test_df_lst=mi_test_df_lst, B=1, y_test=y_test, task_name=task_type)
mi_tsmb_B5 = generate_results_function.tsmb_select_b_func(test_df_lst=mi_test_df_lst, B=5, y_test=y_test, task_name=task_type)
mi_tsmb_B10 = generate_results_function.tsmb_select_b_func(test_df_lst=mi_test_df_lst, B=10, y_test=y_test, task_name=task_type)
mi_tsmb_B20 = generate_results_function.tsmb_select_b_func(test_df_lst=mi_test_df_lst, B=20, y_test=y_test, task_name=task_type)
mi_tsmb_B50 = generate_results_function.tsmb_select_b_func(test_df_lst=mi_test_df_lst, B=50, y_test=y_test, task_name=task_type)
mi_tsmb_B100 = generate_results_function.tsmb_select_b_func(test_df_lst=mi_test_df_lst, B=100, y_test=y_test, task_name=task_type)
mi_tsmb_select_b_lst = [mi_tsmb_B1, mi_tsmb_B5, mi_tsmb_B10, mi_tsmb_B20, mi_tsmb_B50, mi_tsmb_B100]


cc_perturbed_B1 = generate_results_function.tsmb_select_b_func(test_df_lst=p_cc_test_df_lst, B=1, y_test=y_test, task_name=task_type)
cc_perturbed_B5 = generate_results_function.tsmb_select_b_func(test_df_lst=p_cc_test_df_lst, B=5, y_test=y_test, task_name=task_type)
cc_perturbed_B10 = generate_results_function.tsmb_select_b_func(test_df_lst=p_cc_test_df_lst, B=10, y_test=y_test, task_name=task_type)
cc_perturbed_B20 = generate_results_function.tsmb_select_b_func(test_df_lst=p_cc_test_df_lst, B=20, y_test=y_test, task_name=task_type)
cc_perturbed_B50 = generate_results_function.tsmb_select_b_func(test_df_lst=p_cc_test_df_lst, B=50, y_test=y_test, task_name=task_type)
cc_perturbed_B100 = generate_results_function.tsmb_select_b_func(test_df_lst=p_cc_test_df_lst, B=100, y_test=y_test, task_name=task_type)
cc_perturbed_select_b_lst = [cc_perturbed_B1, cc_perturbed_B5, cc_perturbed_B10, cc_perturbed_B20, cc_perturbed_B50, cc_perturbed_B100]
mi_perturbed_B1 = generate_results_function.tsmb_select_b_func(test_df_lst=p_mi_test_df_lst, B=1, y_test=y_test, task_name=task_type)
mi_perturbed_B5 = generate_results_function.tsmb_select_b_func(test_df_lst=p_mi_test_df_lst, B=5, y_test=y_test, task_name=task_type)
mi_perturbed_B10 = generate_results_function.tsmb_select_b_func(test_df_lst=p_mi_test_df_lst, B=10, y_test=y_test, task_name=task_type)
mi_perturbed_B20 = generate_results_function.tsmb_select_b_func(test_df_lst=p_mi_test_df_lst, B=20, y_test=y_test, task_name=task_type)
mi_perturbed_B50 = generate_results_function.tsmb_select_b_func(test_df_lst=p_mi_test_df_lst, B=50, y_test=y_test, task_name=task_type)
mi_perturbed_B100 = generate_results_function.tsmb_select_b_func(test_df_lst=p_mi_test_df_lst, B=100, y_test=y_test, task_name=task_type)
mi_perturbed_select_b_lst = [mi_perturbed_B1, mi_perturbed_B5, mi_perturbed_B10, mi_perturbed_B20, mi_perturbed_B50, mi_perturbed_B100]


cc_tdb_B1 = generate_results_function.tdb_select_b_func(b_lst=b_cc_lst, B=1, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_B5 = generate_results_function.tdb_select_b_func(b_lst=b_cc_lst, B=5, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_B10 = generate_results_function.tdb_select_b_func(b_lst=b_cc_lst, B=10, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_B20 = generate_results_function.tdb_select_b_func(b_lst=b_cc_lst, B=20, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_B50 = generate_results_function.tdb_select_b_func(b_lst=b_cc_lst, B=50, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_B100 = generate_results_function.tdb_select_b_func(b_lst=b_cc_lst, B=100, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
cc_tdb_select_b_lst = [cc_tdb_B1, cc_tdb_B5, cc_tdb_B10, cc_tdb_B20, cc_tdb_B50, cc_tdb_B100]
mi_tdb_B1 = generate_results_function.tdb_select_b_func(b_lst=b_mi_lst, B=1, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_B5 = generate_results_function.tdb_select_b_func(b_lst=b_mi_lst, B=5, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_B10 = generate_results_function.tdb_select_b_func(b_lst=b_mi_lst, B=10, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_B20 = generate_results_function.tdb_select_b_func(b_lst=b_mi_lst, B=20, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_B50 = generate_results_function.tdb_select_b_func(b_lst=b_mi_lst, B=50, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_B100 = generate_results_function.tdb_select_b_func(b_lst=b_mi_lst, B=100, X_training=X_training, y_training=y_training, X_test=X_test, y_test=y_test, task_name=task_type, tolerance=tolerance)
mi_tdb_select_b_lst = [mi_tdb_B1, mi_tdb_B5, mi_tdb_B10, mi_tdb_B20, mi_tdb_B50, mi_tdb_B100]


tsmb_select_b = generate_results_function.make_select_b_df(all_final_cc_lst=cc_tsmb_select_b_lst, all_final_mi_lst=mi_tsmb_select_b_lst, method_name=enums.MethodType.TSMB.value, dataset_name=dataset_name)
# tsmb_select_b['test_values'] = tsmb_select_b['test_values'] - orderly['orderly_test_result_dict']['orderly_no_alignment_indicator']
print("tsmb select b completed")
perturbed_select_b = generate_results_function.make_select_b_df(all_final_cc_lst=cc_perturbed_select_b_lst, all_final_mi_lst=mi_perturbed_select_b_lst, method_name=enums.MethodType.PERTURBED_MODEL.value, dataset_name=dataset_name)
# perturbed_select_b['test_values'] = perturbed_select_b['test_values'] - orderly['orderly_test_result_dict']['orderly_no_alignment_indicator']
print("perturbed select b completed")
tdb_select_b = generate_results_function.make_select_b_df(all_final_cc_lst=cc_tdb_select_b_lst, all_final_mi_lst=mi_tdb_select_b_lst, method_name=enums.MethodType.TDB.value, dataset_name=dataset_name)
# tdb_select_b['test_values'] = tdb_select_b['test_values'] - orderly['orderly_test_result_dict']['orderly_no_alignment_indicator']
print("tdb select b completed")

tsmb_select_b.to_csv(f'{dataset_name}_{data_arrange}_gbdt_tsmb_select_b.csv', index=False)
perturbed_select_b.to_csv(f'{dataset_name}_{data_arrange}_gbdt_perturbed_select_b.csv', index=False)
tdb_select_b.to_csv(f'{dataset_name}_{data_arrange}_gbdt_tdb_select_b.csv', index=False)



## ======  performance  ======
performance_df = generate_results_function.make_performance_df(orderly_dict=orderly, dataset_name=dataset_name,
                             cc_tsmb_performance_lst=cc_tsmb_top100, mi_tsmb_performance_lst=mi_tsmb_top100,
                             cc_perturbed_performance_lst=cc_perturbed_top100, mi_perturbed_performancelst=mi_perturbed_top100,
                             cc_tdb_performance_lst=cc_tdb_top100, mi_tdb_performance_lst=mi_tdb_top100)
# performance_df['test_values'] = performance_df['test_values'] - performance_df[performance_df['Method'] == 'No_alignment'].values[0][1]
performance_idxmax_df = performance_df.iloc[performance_df.groupby(["Method", "run_id"])['test_values'].idxmax()]
sort_list = ['TSMB-TDMI', 'TSMB-GCC', 'Perturbed Model-TDMI', 'Perturbed Model-GCC', 'TDB-TDMI', 'TDB-GCC', 'TDMI', 'GCC', 'No_alignment']
performance_idxmax_df.index = performance_idxmax_df['Method']
final_performance_df = performance_idxmax_df.loc[sort_list]
final_performance_df.reset_index(drop=True, inplace=True)

all_performance_df = final_performance_df[final_performance_df['Method'].isin(['TSMB-TDMI', 'TSMB-GCC', 'Perturbed Model-TDMI', 'Perturbed Model-GCC', 'TDB-TDMI', 'TDB-GCC', 'TDMI', 'GCC'])]
print("all performance completed")
performance_df = final_performance_df[final_performance_df['Method'].isin(['TSMB-TDMI', 'TSMB-GCC', 'TDMI', 'GCC'])]
print("performance completed")

all_performance_df.to_csv(f'{dataset_name}_{data_arrange}_tsmb_all_performance.csv', index=False)
performance_df.to_csv(f'{dataset_name}_{data_arrange}_tsmb_performance.csv', index=False)


# ======  tft  ======
## ===  tft performance  ===
cc_tft_performance = generate_results_function.tft_func(tft_df_lst=tft_cc_test_df_lst, task_name=task_type)
mi_tft_performance = generate_results_function.tft_func(tft_df_lst=tft_mi_test_df_lst, task_name=task_type)
tft_performance_df = generate_results_function.make_tft_performance_df(tft_orderly_dict=tft_orderly, cc_tft_lst=cc_tft_performance, mi_tft_lst=mi_tft_performance, dataset_name=dataset_name)
tft_performance_df.to_csv(f'{dataset_name}_{data_arrange}_tft_tsmb_performance.csv', index=False)
print("tft performance completed")

## ===  tft select b  ===
cc_tft_tsmb_B1 = generate_results_function.tft_select_b_func(test_df_lst=tft_cc_test_df_lst, B=1, y_test=y_test, task_name=task_type)
cc_tft_tsmb_B5 = generate_results_function.tft_select_b_func(test_df_lst=tft_cc_test_df_lst, B=5, y_test=y_test, task_name=task_type)
cc_tft_tsmb_B10 = generate_results_function.tft_select_b_func(test_df_lst=tft_cc_test_df_lst, B=10, y_test=y_test, task_name=task_type)
cc_tft_tsmb_B20 = generate_results_function.tft_select_b_func(test_df_lst=tft_cc_test_df_lst, B=20, y_test=y_test, task_name=task_type)
cc_tft_tsmb_B50 = generate_results_function.tft_select_b_func(test_df_lst=tft_cc_test_df_lst, B=50, y_test=y_test, task_name=task_type)
cc_tft_tsmb_B100 = generate_results_function.tft_select_b_func(test_df_lst=tft_cc_test_df_lst, B=100, y_test=y_test, task_name=task_type)
cc_tft_tsmb_select_b_lst = [cc_tft_tsmb_B1, cc_tft_tsmb_B5, cc_tft_tsmb_B10, cc_tft_tsmb_B20, cc_tft_tsmb_B50, cc_tft_tsmb_B100]

mi_tft_tsmb_B1 = generate_results_function.tft_select_b_func(test_df_lst=tft_mi_test_df_lst, B=1, y_test=y_test, task_name=task_type)
mi_tft_tsmb_B5 = generate_results_function.tft_select_b_func(test_df_lst=tft_mi_test_df_lst, B=5, y_test=y_test, task_name=task_type)
mi_tft_tsmb_B10 = generate_results_function.tft_select_b_func(test_df_lst=tft_mi_test_df_lst, B=10, y_test=y_test, task_name=task_type)
mi_tft_tsmb_B20 = generate_results_function.tft_select_b_func(test_df_lst=tft_mi_test_df_lst, B=20, y_test=y_test, task_name=task_type)
mi_tft_tsmb_B50 = generate_results_function.tft_select_b_func(test_df_lst=tft_mi_test_df_lst, B=50, y_test=y_test, task_name=task_type)
mi_tft_tsmb_B100 = generate_results_function.tft_select_b_func(test_df_lst=tft_mi_test_df_lst, B=100, y_test=y_test, task_name=task_type)
mi_tft_tsmb_select_b_lst = [mi_tft_tsmb_B1, mi_tft_tsmb_B5, mi_tft_tsmb_B10, mi_tft_tsmb_B20, mi_tft_tsmb_B50, mi_tft_tsmb_B100]

tft_tsmb_select_b = generate_results_function.make_select_b_df(all_final_cc_lst=cc_tft_tsmb_select_b_lst, all_final_mi_lst=mi_tft_tsmb_select_b_lst, method_name=enums.MethodType.TSMB.value, dataset_name=dataset_name)
# tft_tsmb_select_b['test_values'] = tft_tsmb_select_b['test_values'] - tft_orderly['orderly_tft_no_alignment_indicator']
tft_tsmb_select_b.to_csv(f'{dataset_name}_{data_arrange}_tft_select_b.csv', index=False)
print("tdt select b completed")