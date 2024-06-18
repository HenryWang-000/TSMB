from enum import Enum


class OptType(Enum):
    ORDERLY= "orderly"
    BOOTSTRAP = "bootstrap"


class ModelDataType(Enum):
    TRAINING = "training"
    TEST = "test"
    TRAIN = "train"
    VAL = "val"


class ToleranceType(Enum):
    MINUTE = "1min"
    HOUR = "1h"
    TWELVEHOURS = "12h"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"



class MethodType(Enum):
    TSMB = "TSMB"
    PERTURBED_MODEL = "Perturbed Model"
    TDB = "TDB"


class CorrMethod(Enum):
    GCC= "cc"
    TDMI = "mi"
    CLASSIF_MI = "classif_mi"
    TSMB_GCC = "TSMB-GCC"
    TSMB_TDMI = "TSMB-TDMI"


class DataSetName(Enum):
    AIR_QUALITY= "air_quality"
    OCCUPANCY = "occupancy"
    PUMP_SENSOR = "pump_sensor"
    POWER_DEMAND = "power_demand"


class DataArrange(Enum):
    FIXED= "fixed"
    STOCHASTIC = "stochastic"