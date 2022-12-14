import os.path
import pathlib

import torch

# Torch device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# Data location constant
PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = os.path.join(PATH, "data")

# Data quality constants
FIPS_BAD_QUALITY_SOIL_DATA = [
    6075,
    6083,
    6087,
    9009,
    12009,
    12037,
    13127,
    13191,
    22023,
    22045,
    22051,
    22075,
    22087,
    22099,
    23009,
    23013,
    23015,
    24047,
    25007,
    25009,
    25019,
    25025,
    28047,
    34009,
    34017,
    36047,
    36059,
    36081,
    36085,
    36103,
    37013,
    37019,
    37029,
    37031,
    37055,
    37095,
    37129,
    37137,
    37139,
    37143,
    44005,
    44009,
    45013,
    45019,
    45043,
    48167,
    51131,
    51810,
    53055,
]

FIPS_NOT_IN_TIMESERIES = [56045]

FIPS_TO_DROP = FIPS_BAD_QUALITY_SOIL_DATA + FIPS_NOT_IN_TIMESERIES
