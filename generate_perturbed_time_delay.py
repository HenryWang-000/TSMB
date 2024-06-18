import numpy as np
import pickle
import os
import model_utils
import opt

from model_utils import *

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


# ========= Import opt results ==========
o_cc_pkl = open(f'orderly_{data_arrange}_training_col{col_nums}_cc_test1.pkl', 'rb')
o_cc = pickle.load(o_cc_pkl)
o_mi_pkl = open(f'orderly_{data_arrange}_training_col{col_nums}_{mi_type}_test1.pkl', 'rb')
o_mi = pickle.load(o_mi_pkl)





def perburbed(task_type, noise_rate, cc_decimal_delta, cc_decimal_window_size,
                            mi_decimal_delta, mi_decimal_window_size,
                            X_training, y_training, X_train, y_train, X_val, y_val, X_test, y_test, dicts, tolerance):
    perturbed_orderly_test_cc_indicator_lst = []
    perturbed_orderly_val_cc_indicator_lst = []
    perturbed_cc_delta_lst = []
    perturbed_cc_window_size_lst = []

    perturbed_orderly_test_mi_indicator_lst = []
    perturbed_orderly_val_mi_indicator_lst = []
    perturbed_mi_delta_lst = []
    perturbed_mi_window_size_lst = []


    for j in range(constants.BOOTSTRAP_NUMS):
        rng = np.random.default_rng()
        d = rng.standard_normal(len(cc_decimal_delta)) * noise_rate
        w = rng.standard_normal(1) * noise_rate

        perturbed_cc_decimal_delta = cc_decimal_delta + d
        perturbed_cc_decimal_window_size = float(cc_decimal_window_size + w)
        perturbed_cc_delta = opt_utils.reduction(arr=perturbed_cc_decimal_delta,
                                          dicts = dicts, key='delta')
        perturbed_cc_window_size = opt_utils.scale(arr=perturbed_cc_decimal_window_size,
                                            dicts = dicts, key='window_size')
        perturbed_cc_delta_lst.append(perturbed_cc_delta)
        perturbed_cc_window_size_lst.append(perturbed_cc_window_size)

        test_cc_results = model_utils.orderly_model(
            task_type=task_type, val_or_test=enums.ModelDataType.TEST.value,
            delta=perturbed_cc_delta,
            window_size=perturbed_cc_window_size,
            X_training=X_training, y_training=y_training,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, tolerance=tolerance)
        val_cc_results = model_utils.orderly_model(
            task_type=task_type, val_or_test=enums.ModelDataType.VAL.value,
            delta=perturbed_cc_delta,
            window_size=perturbed_cc_window_size,
            X_training=X_training, y_training=y_training,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, tolerance=tolerance)
        perturbed_orderly_test_cc_indicator_lst.append(test_cc_results[0])
        perturbed_orderly_val_cc_indicator_lst.append(val_cc_results[0])

        print(f'Perturbed_cc_{j} completed')

        #####################################################################################################

        perturbed_mi_decimal_delta = mi_decimal_delta + d
        perturbed_mi_decimal_window_size = float(mi_decimal_window_size + w)
        perturbed_mi_delta = opt_utils.reduction(arr=perturbed_mi_decimal_delta,
                                             dicts = dicts, key='delta')
        perturbed_mi_window_size = opt_utils.scale(arr=perturbed_mi_decimal_window_size,
                                               dicts = dicts, key='window_size')
        perturbed_mi_delta_lst.append(perturbed_mi_delta)
        perturbed_mi_window_size_lst.append(perturbed_mi_window_size)

        test_mi_results = model_utils.orderly_model(
            task_type=task_type, val_or_test=enums.ModelDataType.TEST.value,
            delta=perturbed_mi_delta,
            window_size=perturbed_mi_window_size,
            X_training=X_training, y_training=y_training,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, tolerance=tolerance)
        val_mi_results = model_utils.orderly_model(
            task_type=task_type, val_or_test=enums.ModelDataType.VAL.value,
            delta=perturbed_mi_delta,
            window_size=perturbed_mi_window_size,
            X_training=X_training, y_training=y_training,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, tolerance=tolerance)
        perturbed_orderly_test_mi_indicator_lst.append(test_mi_results[0])
        perturbed_orderly_val_mi_indicator_lst.append(val_mi_results[0])

        print(f'Perturbed_mi_{j} completed')
        print(f'The {j} perturbed has been completed')
        print('\n')

    return perturbed_cc_delta_lst, perturbed_cc_window_size_lst, \
           perturbed_orderly_test_cc_indicator_lst, perturbed_orderly_val_cc_indicator_lst, \
           perturbed_mi_delta_lst, perturbed_mi_window_size_lst, \
           perturbed_orderly_test_mi_indicator_lst, perturbed_orderly_val_mi_indicator_lst



times = 1
while times <= constants.BOOTSTRAP_ITERATIONS:
    cc_d, cc_w, cc_t_indicator, cc_v_indicator, mi_d, mi_w, mi_t_indicator, mi_v_indicator = perburbed(
        task_type=task_type,
        noise_rate=constants.NOISE_RATE,
        cc_decimal_delta=o_cc['decimal_delta'],
        cc_decimal_window_size=o_cc['decimal_window_size'],
        mi_decimal_delta=o_mi['decimal_delta'],
        mi_decimal_window_size=o_mi['decimal_window_size'],
        X_training=X_training, y_training=y_training,
        X_train=X_training, y_train=y_training,
        X_val=X_test, y_val=y_test,
        X_test=X_test, y_test=y_test, dicts=dicts, tolerance=tolerance
    )

    perturbed_result_dict = {
        'cc_perturbed_delta': cc_d,
        'cc_perturbed_window_size': cc_w,
        'cc_perturbed_test_indicator': cc_t_indicator,
        'cc_perturbed_val_indicator': cc_v_indicator,
        'mi_perturbed_delta': mi_d,
        'mi_perturbed_window_size': mi_w,
        'mi_perturbed_test_indicator': mi_t_indicator,
        'mi_perturbed_val_indicator': mi_v_indicator
    }

    file = f'{dataset_name}_{data_arrange}_perturbed_result_test{times}.pkl'
    if not os.path.exists(file):
        with open(file=file, mode="wb") as f:
            pickle.dump(perturbed_result_dict, f, True)
        times += 1


