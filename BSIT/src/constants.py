"""Here we put all the constants that we use across our module.

Some examples are paths to data or to figures folders.

.. note::
    If you notice that you often change any of the constants, than it is not a
    constant and it does not belong here.
"""
"""Here we put all the constants that we use across our module.

Some examples are paths to data or to figures folders.

.. note::
    If you notice that you often change any of the constants, than it is not a
    constant and it does not belong here.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    root = Path("data")
    raw = root / "raw"
    processed = root / "processed"


FIGURES_PATH = Path("reports/figures")
FREQUENCY = 250
NEW_FREQUENCY = 200
DOWN_FREQUENCY = 20

VALID_IDX = [10067,10034,6966]
TEST_IDX = [10034,10006,10012,10033,3973]
TRAIN_IDX = [10013,10012,10016,10019,10042,10089,10018,10024,10023,10039,10018,10022,10000,10037,10082,10033,10020,10018,10030,1282]

INCLUDED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']



# @dataclass
# class DataPaths:
#     root = Path("data")
#     raw = root / "raw"
#     processed = root / "processed"


# FIGURES_PATH = Path("reports/figures")
