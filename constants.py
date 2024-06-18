# Constants

RAMDOM_SEED = 42
BOOTSTRAP_NUMS = 100
BOOTSTRAP_ITERATIONS = 5
NOISE_RATE = 0.1

## Time Delay Range
OCCUPANCY_TD_RANGE = {'delta': (10.0, 180.0), 'window_size': (1.0, 60.0)}
PUMP_SENSOR_TD_RANGE = {'delta': (10.0, 80.0), 'window_size': (1.0, 150.0)}
AIR_QUALITY_TD_RANGE = {'delta': (60.0, 1440.0), 'window_size': (1.0, 10800.0)}
POWER_DEMAND_TD_RANGE = {'delta': (0.0, 10.0), 'window_size': (1.0, 1.0)}



## TFT Hyper Parameterconda
USE_GPU = 1
MAX_ENCODER_LENGTH = 1000
BATCH_SIZE = 32
MAX_EPOCHS = 30
GRADIENT_CLIP_VAL = 0.1
LIMIT_TRAIN_BATCHES = 30
LEARNING_RATE = 3.0E-3
HIDDEN_SIZE = 20
ATTENTION_HEAD_SIZE = 1
DROUPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 8
CLASSIFICATION_OUTPUT_SIZE = 2
REGRESSION_OUTPUT_SIZE = 1
REDUCE_ON_PLATEAU_PATIENCE = 4



## Features_Target Dicts
OCCUPANCY_DICTS = {"features":['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'], "target":"Occupancy"}
PUMP_SENSOR_DICTS = {"features":['sensor_04', 'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09', 'sensor_10'], "target":"machine_status"}
AIR_QUALITY_DICTS = {"features":['PT08S1(CO)', 'PT08S2(NMHC)', 'PT08S3(NOx)', 'PT08S4(NO2)', 'PT08S5(O3)', 'T', 'RH', 'AH'], "target":"CO(GT)"}
POWER_DEMAND_DICTS = {"features":['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13', 'att14',
                                  'att15', 'att16', 'att17', 'att18', 'att19', 'att20', 'att21', 'att22', 'att23', 'att24'], "target":"target"}
AIR_QUALITY_FEATURES_MAPPING = {
    'PT08.S1(CO)': 'PT08S1(CO)',
    'PT08.S2(NMHC)': 'PT08S2(NMHC)',
    'PT08.S3(NOx)': 'PT08S3(NOx)',
    'PT08.S4(NO2)': 'PT08S4(NO2)',
    'PT08.S5(O3)': 'PT08S5(O3)'
}