import numpy as np
import pandas as pd
import datetime
import pytorch_forecasting
import constants
import opt_utils
import enums
import itertools
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from sklearn.metrics import r2_score
from functools import reduce
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore", message="`trainer.fit(train_dataloader)` is deprecated.*")

seed = constants.RAMDOM_SEED

def gdbt_regressor_model(train, test):
    forest = GradientBoostingRegressor(random_state=seed)
    forest.fit(train.iloc[:, 1:-1], train.iloc[:, -1])

    score = forest.score(test.iloc[:, 1:-1], test.iloc[:, -1])
    test_predict = forest.predict(test.iloc[:, 1:-1])
    rmse = metrics.mean_squared_error(test.iloc[:, -1], test_predict) ** 0.5

    return score, rmse, test_predict


def gdbt_classifier_model(train, test):
    rfc = GradientBoostingClassifier(random_state=seed)
    rfc.fit(train.iloc[:, 1:-1], train.iloc[:, -1])
    y_pred = rfc.predict(test.iloc[:, 1:-1])
    fpr, tpr, thresholds = metrics.roc_curve(test.iloc[:, -1], y_pred)
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(test.iloc[:, -1], y_pred)
    f1 = metrics.f1_score(test.iloc[:, -1], y_pred)
    y_pred_proba = rfc.predict_proba(test.iloc[:, 1:-1])

    return auc, accuracy, f1, y_pred_proba


def orderly_model(task_type, val_or_test, delta, window_size, X_training, y_training, X_train, y_train, X_val, y_val, X_test, y_test, tolerance):
    def transform_and_model(X1, y1, X2, y2, model_func):
        data, data_test = opt_utils.transform_data(delta=delta, window_size=window_size,
                                                               X_training=X1,
                                                               y_training=y1, X_test=X2, y_test=y2, tolerance=tolerance)
        return model_func(data, data_test)

    model_functions = {
        "regression": (gdbt_regressor_model, (X_training, y_training, X_test, y_test), (X_train, y_train, X_val, y_val)),
        "classification": (gdbt_classifier_model, (X_training, y_training, X_test, y_test), (X_train, y_train, X_val, y_val))
    }

    if task_type in model_functions:
        model_func, test_args, val_args = model_functions[task_type]
        if val_or_test == 'test':
            return transform_and_model(*test_args, model_func)
        elif val_or_test == 'val':
            return transform_and_model(*val_args, model_func)
        else:
            raise RuntimeError(f'cannot find val_or_test: {val_or_test}')
    else:
        raise RuntimeError(f'cannot find task_type: {task_type}')




def bootstrap_model(task_type, val_or_test, pkl, X_training, y_training, X_train, y_train, X_val, y_val, X_test, y_test, tolerance):
    if val_or_test not in [enums.ModelDataType.TEST.value, enums.ModelDataType.VAL.value]:
        raise RuntimeError(f'cannot find val_or_test: {val_or_test}')

    df_lst = []
    data_X, data_y = (X_training, y_training) if val_or_test == enums.ModelDataType.TEST.value else (X_train, y_train)
    data_test_X, data_test_y = (X_test, y_test) if val_or_test == enums.ModelDataType.TEST.value else (X_val, y_val)

    for i in range(constants.BOOTSTRAP_NUMS):
        data, data_test = opt_utils.transform_data(delta=pkl['bootstrap_process'][i]['delta'],
                                   window_size=pkl['bootstrap_process'][i]['window_size'],
                                   X_training=data_X, y_training=data_y, X_test=data_test_X, y_test=data_test_y, tolerance=tolerance)

        if task_type == enums.TaskType.CLASSIFICATION.value:
            auc, accuracy, f1, y_pred_proba = gdbt_classifier_model(data, data_test)
            df = pd.DataFrame({'time': data_test.iloc[:, 0],
                               f'{val_or_test}_y_pred_proba0_{i}': y_pred_proba.T[0],
                               f'{val_or_test}_y_pred_proba1_{i}': y_pred_proba.T[1],
                               f'{val_or_test}_auc_{i}': [auc] * len(data_test),
                               f'{val_or_test}_accuracy_{i}': [accuracy] * len(data_test),
                               f'{val_or_test}_f1_{i}': [f1] * len(data_test)})
        elif task_type == enums.TaskType.REGRESSION.value:
            score, rmse, predict = gdbt_regressor_model(data, data_test)
            df = pd.DataFrame({'time': data_test.iloc[:, 0], f'{val_or_test}_y_pred_{i}': predict,
                               f'{val_or_test}_r2_{i}': [score] * len(data_test),
                               f'{val_or_test}_rmse_{i}': [rmse] * len(data_test)})
        else : df = pd.DataFrame()
        df_lst.append(df)

    max_df = reduce(lambda left, right: pd.merge(left, right, on=['time'], how='outer'), df_lst)

    return max_df


def get_bootstrap_results(task_type, pkl_lst, X_training, y_training, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, method_name, data_arrange, tolerance):
    all_test_max_df_lst = []
    all_val_max_df_lst = []

    for j in range(len(pkl_lst)):
        test_max_df = bootstrap_model(task_type=task_type, val_or_test=enums.ModelDataType.TEST.value, pkl=pkl_lst[j],
                                                  X_training=X_training, y_training=y_training,
                                                  X_train=X_train, y_train=y_train,
                                                  X_val=X_val, y_val=y_val,
                                                  X_test=X_test, y_test=y_test, tolerance=tolerance)
        val_max_df = bootstrap_model(task_type=task_type, val_or_test=enums.ModelDataType.VAL.value, pkl=pkl_lst[j],
                                                 X_training=X_training, y_training=y_training,
                                                 X_train=X_train, y_train=y_train,
                                                 X_val=X_val, y_val=y_val,
                                                 X_test=X_test, y_test=y_test, tolerance=tolerance)


        test_max_df.to_pickle(f"{data_arrange}_{dataset_name}_{method_name}_test_max_df_{j}.pkl")
        val_max_df.to_pickle(f"{data_arrange}_{dataset_name}_{method_name}_val_max_df_{j}.pkl")

        all_test_max_df_lst.append((test_max_df))
        all_val_max_df_lst.append(val_max_df)

        print(f'The {j} th record of bootstrap result_dict has been completed')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    all_test_max_df = pd.concat(all_test_max_df_lst, axis=0)
    all_val_max_df = pd.concat(all_val_max_df_lst, axis=0)

    return all_test_max_df, all_val_max_df



def perturbed_bootstrap_model(task_type, val_or_test, delta_lst, window_size_lst, X_training, y_training, X_train, y_train, X_val, y_val, X_test, y_test, tolerance):
    if val_or_test not in [enums.ModelDataType.TEST.value, enums.ModelDataType.VAL.value]:
        raise RuntimeError(f'cannot find val_or_test: {val_or_test}')

    df_lst = []
    data_X, data_y = (X_training, y_training) if val_or_test == enums.ModelDataType.TEST.value else (X_train, y_train)
    data_test_X, data_test_y = (X_test, y_test) if val_or_test == enums.ModelDataType.TEST.value else (X_val, y_val)

    for i in range(constants.BOOTSTRAP_NUMS):
        data, data_test = opt_utils.transform_data(delta=delta_lst[i], window_size=window_size_lst[i],
                                                   X_training=data_X, y_training=data_y, X_test=data_test_X, y_test=data_test_y, tolerance=tolerance)

        if task_type == enums.TaskType.CLASSIFICATION.value:
            auc, accuracy, f1, y_pred_proba = gdbt_classifier_model(data, data_test)
            df = pd.DataFrame({'time': data_test.iloc[:, 0],
                               f'{val_or_test}_y_pred_proba0_{i}': y_pred_proba.T[0],
                               f'{val_or_test}_y_pred_proba1_{i}': y_pred_proba.T[1],
                               f'{val_or_test}_auc_{i}': [auc] * len(data_test),
                               f'{val_or_test}_accuracy_{i}': [accuracy] * len(data_test),
                               f'{val_or_test}_f1_{i}': [f1] * len(data_test)})
        elif task_type == enums.TaskType.REGRESSION.value:
            score, rmse, predict = gdbt_regressor_model(data, data_test)
            df = pd.DataFrame({'time': data_test.iloc[:, 0], f'test_y_pred_{i}': predict,
                               f'test_r2_{i}': [score] * len(data_test),
                               f'test_rmse_{i}': [rmse] * len(data_test)})
        else : df = pd.DataFrame()
        df_lst.append(df)

    max_df = reduce(lambda left, right: pd.merge(left, right, on=['time'], how='outer'), df_lst)

    return max_df


def get_perturbed_bootstrap_results(task_type, delta_pkl_lst, window_size_pkl_lst, X_training, y_training, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, method_name, data_arrange, tolerance):
    all_test_max_df_lst = []
    all_val_max_df_lst = []

    for j in range(len(delta_pkl_lst)):
        test_max_df = perturbed_bootstrap_model(task_type=task_type, val_or_test=enums.ModelDataType.TEST.value,
                                      delta_lst=delta_pkl_lst[j], window_size_lst=window_size_pkl_lst[j],
                                      X_training=X_training, y_training=y_training, X_train=X_train, y_train=y_train,
                                      X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, tolerance=tolerance)
        val_max_df = perturbed_bootstrap_model(task_type=task_type, val_or_test=enums.ModelDataType.VAL.value,
                                     delta_lst=delta_pkl_lst[j], window_size_lst=window_size_pkl_lst[j],
                                     X_training=X_training, y_training=y_training, X_train=X_train, y_train=y_train,
                                     X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, tolerance=tolerance)


        test_max_df.to_pickle(f"p_{data_arrange}_{dataset_name}_{method_name}_test_max_df_{j}.pkl")
        val_max_df.to_pickle(f"p_{data_arrange}_{dataset_name}_{method_name}_val_max_df_{j}.pkl")

        all_test_max_df_lst.append((test_max_df))
        all_val_max_df_lst.append(val_max_df)

        print(f'The {j} th record of bootstrap result_dict has been completed')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    all_test_max_df = pd.concat(all_test_max_df_lst, axis=0)
    all_val_max_df = pd.concat(all_val_max_df_lst, axis=0)

    return all_test_max_df, all_val_max_df



def tft_model(task_type, df, train_cutoff, test_cutoff, data_dicts):
    torch.cuda.empty_cache()
    tft_df = df.copy()
    tft_df['group'] = 0
    tft_df['date_block_num'] = tft_df.index
    max_prediction_length = test_cutoff
    max_encoder_length = train_cutoff
    loss = pytorch_forecasting.metrics.CrossEntropy() if task_type == enums.TaskType.CLASSIFICATION.value else pytorch_forecasting.metrics.RMSE()
    output_size = constants.CLASSIFICATION_OUTPUT_SIZE if task_type == enums.TaskType.CLASSIFICATION.value else constants.REGRESSION_OUTPUT_SIZE
    data = pd.DataFrame()

    if data_dicts == constants.PUMP_SENSOR_DICTS:
        data = tft_df.copy()
        tft_df = data.iloc[:train_cutoff + 10000]
        max_prediction_length = 10000


    training = TimeSeriesDataSet(
        tft_df[:train_cutoff],
        time_idx='date_block_num',
        target=data_dicts["target"],
        group_ids=['group'],
        min_encoder_length=0,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=data_dicts["features"],
        time_varying_unknown_reals=[data_dicts["target"]],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, tft_df, predict=True, stop_randomization=True)
    batch_size = constants.BATCH_SIZE
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    pl.seed_everything(constants.RAMDOM_SEED)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=3, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=constants.MAX_EPOCHS,
        gpus=constants.USE_GPU,
        weights_summary="top",
        gradient_clip_val=constants.GRADIENT_CLIP_VAL,
        limit_train_batches=constants.LIMIT_TRAIN_BATCHES,
        callbacks=[lr_logger, early_stop_callback]
    )
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=constants.LEARNING_RATE,
        hidden_size=constants.HIDDEN_SIZE,
        attention_head_size=constants.ATTENTION_HEAD_SIZE,
        dropout=constants.DROUPOUT,
        hidden_continuous_size=constants.HIDDEN_CONTINUOUS_SIZE,
        output_size=output_size,
        loss=loss,
        reduce_on_plateau_patience=constants.REDUCE_ON_PLATEAU_PATIENCE,
    )
    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print("path: ", best_model_path)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    if data_dicts == constants.PUMP_SENSOR_DICTS:
        preds_lst = []
        for j in range(10):
            df_test = data[train_cutoff + j * 10000:train_cutoff + (j + 1) * 10000]
            pred, x = best_tft.predict(df_test, return_x=True)
            preds_lst.append(list(np.array(pred[0])))

        all_preds = list(itertools.chain.from_iterable(preds_lst))
        fpr, tpr, thresholds = metrics.roc_curve(data.loc[:, constants.PUMP_SENSOR_DICTS["target"]][train_cutoff:train_cutoff + 100000],
                                                 all_preds)
        return data[train_cutoff:train_cutoff + 100000], all_preds, metrics.auc(fpr, tpr)

    else:
        df_test = tft_df[train_cutoff:]
        pred_lst, x = best_tft.predict(df_test, return_x=True)
        preds = pred_lst[0]

        if task_type == enums.TaskType.CLASSIFICATION.value:
            fpr, tpr, thresholds = metrics.roc_curve(df_test.iloc[:, -3], preds)
            return df_test, preds, metrics.auc(fpr, tpr)
        elif task_type == enums.TaskType.REGRESSION.value:
            return df_test, preds, r2_score(df_test.iloc[:, -3], preds)
        else:
            raise RuntimeError(f'cannot find task_type: {task_type}')


def single_tft_model(task_type, data_dicts, tolerance, delta=None, window_size=None, train_df=None, test_df=None, X_df=None, y_df=None, train_X_df=None, train_y_df=None, test_X_df=None, test_y_df=None):

    if X_df is not None and y_df is not None:
        df = opt_utils.transform(delta=delta, window_size=window_size, X_df=X_df, y_df=y_df, tolerance=tolerance)
        train_cutoff = int(len(df) * 0.5)
        test_cutoff = int(len(df) * 0.75)
    elif train_X_df is not None and train_y_df is not None and test_X_df is not None and test_y_df is not None:
        train_df, test_df = opt_utils.transform_data(delta=delta, window_size=window_size, X_training=train_X_df,
                                                   y_training=train_y_df, X_test=test_X_df, y_test=test_y_df, tolerance=tolerance)
        df = pd.concat([train_df, test_df])
        train_cutoff = int(len(train_df))
        test_cutoff = int(len(test_df))
    elif train_df is not None and test_df is not None:
        df = pd.concat([train_df, test_df])
        train_cutoff = int(len(train_df))
        test_cutoff = int(len(test_df))
    else:
        raise RuntimeError("Invalid arguments")

    df.reset_index(inplace=True, drop=True)
    if data_dicts == constants.AIR_QUALITY_DICTS:
        df.rename(columns=constants.AIR_QUALITY_FEATURES_MAPPING, inplace=True)
    df_test, preds, indicator = tft_model(task_type=task_type, df=df, train_cutoff=train_cutoff, test_cutoff=test_cutoff, data_dicts=data_dicts)

    return indicator



def bootstrap_tft_model(task_type, data_dicts, bootstrap_td_lst, tolerance, X_df=None, y_df=None, train_X_df=None, train_y_df=None, test_X_df=None, test_y_df=None):
    df_lst = []

    for i in range(constants.BOOTSTRAP_NUMS):
        if X_df is not None and y_df is not None:
            df = opt_utils.transform(delta=bootstrap_td_lst['bootstrap_process'][i]['delta'],
                                 window_size=bootstrap_td_lst['bootstrap_process'][i]['window_size'],
                                 X_df=X_df, y_df=y_df, tolerance=tolerance)
            train_cutoff = int(len(df) * 0.5)
            test_cutoff = int(len(df) * 0.75)
        elif train_X_df is not None and train_y_df is not None and test_X_df is not None and test_y_df is not None:
            train_df, test_df = opt_utils.transform_data(delta=bootstrap_td_lst['bootstrap_process'][i]['delta'],
                                           window_size=bootstrap_td_lst['bootstrap_process'][i]['window_size'],
                                           X_training=train_X_df, y_training=train_y_df, X_test=test_X_df, y_test=test_y_df, tolerance=tolerance)

            df = pd.concat([train_df, test_df])
            train_cutoff = len(train_df)
            test_cutoff = len(test_df)
        else:
            raise RuntimeError("Invalid arguments")

        df.reset_index(inplace=True, drop=True)
        if data_dicts == constants.AIR_QUALITY_DICTS:
            df.rename(columns=constants.AIR_QUALITY_FEATURES_MAPPING, inplace=True)
        df_test, preds, indicator = tft_model(task_type=task_type, df=df, train_cutoff=train_cutoff, test_cutoff=test_cutoff, data_dicts=data_dicts)

        if task_type == enums.TaskType.CLASSIFICATION.value:
            result_df = pd.DataFrame(
                {'time': df_test.iloc[:, 0], f'y_true_{i}': df_test.iloc[:, -3], f'y_pred_{i}': preds,
                 f'auc_{i}': indicator})
        elif task_type == enums.TaskType.REGRESSION.value:
            result_df = pd.DataFrame(
                {'time': df_test.iloc[:, 0], f'y_true_{i}': df_test.iloc[:, -3], f'y_pred_{i}': preds,
                 f'r2_{i}': indicator})
        else:
            raise RuntimeError(f'cannot find task_type: {task_type}')

        df_lst.append(result_df)
        print(f'The {i} th tft completion')

    max_df = reduce(lambda left, right: pd.merge(left, right, on=['time'], how='outer'), df_lst)

    return max_df


def make_orderly_tft_result_dict(cc_indicator, mi_indicator, no_alignment_indicator, real_time_delay_indicator):
    orderly_tft_result_dict = {'orderly_tft_cc_indicator': cc_indicator,
                               'orderly_tft_mi_indicator': mi_indicator,
                               'orderly_tft_no_alignment_indicator': no_alignment_indicator,
                               'orderly_real_time_delay_indicator': real_time_delay_indicator}
    return orderly_tft_result_dict
