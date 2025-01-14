""" Everything related to CHUV metadata.

For examples you can find:

* training sessions
* testing sessions

Notes
-----
Training sessions are all the sessions from session ``UP2001_2023_10_23_BSITraining_day3_PhysioUpperLead\WIMAGINE``
that contain some ECoG data.


*Author: Etienne de Montalivet*
"""

import os
from pathlib import Path

from loguru import logger

UP2001_TESTING_SESSIONS = [
    "UP2001_2023_11_01_BSITraining_day10",
    "UP2001_2023_11_14_BSITraining_day18",
    "UP2001_2023_11_24_Rehabilitation_day4_Reaching and hand",
    "UP2001_2023_12_06_Rehabilitation_day11",
    "UP2001_2023_12_15_Rehabilitation_day17_Rehab_6states",
    "UP2001_2024_01_10_Rehabilitation_day24",
    "UP2001_2024_01_31_Rehabilitation_day38",
    "UP2001_2024_02_12_Rehabilitation_day46",
]

UP2001_TRAINING_SESSIONS = [
    "UP2001_2023_10_23_BSITraining_day3_PhysioUpperLead",
    "UP2001_2023_10_24_BSITraining_day4",
    "UP2001_2023_10_25_BSITraining_day5",
    "UP2001_2023_10_26_BSITraining_day6",
    "UP2001_2023_10_27_BSITraining_day7",
    "UP2001_2023_10_30_BSITraining_day8",
    "UP2001_2023_11_02_BSITraining_day11",
    "UP2001_2023_11_03_BSITraining_day12_ALExBrainReward",
    "UP2001_2023_11_06_BSITraining_day13",
    "UP2001_2023_11_07_BSITraining_day14_2states",
    "UP2001_2023_11_08_BSITraining_day15",
    "UP2001_2023_11_10_BSITraining_day16_4states",
    "UP2001_2023_11_13_BSITraining_day17_4states",
    "UP2001_2023_11_15_BSITraining_Day19",
    "UP2001_2023_11_16_BSITraining_day20",
    "UP2001_2023_11_17_BSITraining_day21",
    "UP2001_2023_11_21_Rehabilitation_day1_hand",
    "UP2001_2023_11_22_Rehabilitation_day2_lowerLeadmapping",
    "UP2001_2023_11_23_Rehabilitation_day3_Hand",
    "UP2001_2023_11_29_Rehabilitation_day6",
    "UP2001_2023_11_30_Rehabilitation_day7",
    "UP2001_2023_12_01_Rehabilitation_day8",
    "UP2001_2023_12_04_Rehabilitation_day9_BrainGPT",
    "UP2001_2023_12_05_Rehabilitation_day10",
    "UP2001_2023_12_08_Rehabilitation_day13",
    "UP2001_2023_12_11_Rehabilitation_day14_BrainGPTstim",
    "UP2001_2023_12_13_Rehabilitation_day16_optiOpen_RehabIkr",
    "UP2001_2023_12_18_Rehabilitation_day18_Rehab",
    "UP2001_2023_12_20_Rehabilitation_day19_Rehab",
    "UP2001_2023_12_21_Rehabitilitation_day20_Rehab",
    "UP2001_2023_12_22_Rehabiltation_day21_rehab",
    "UP2001_2024_01_08_Rehabilitation_day22_rehab",
    "UP2001_2024_01_09_Rehabilitation_day23",
    "UP2001_2024_01_11_Rehabilitation_day25",
    "UP2001_2024_01_12_Rehabilitation_day26",
    "UP2001_2024_01_15_Rehabilitation_day27",
    "UP2001_2024_01_16_Rehabilitation_day28",
    "UP2001_2024_01_17_Rehabilitation_day29",
    "UP2001_2024_01_18_Rehabilitation_day30",
    "UP2001_2024_01_19_Rehabilitation_day31",
    "UP2001_2024_01_22_Rehabiltation_day32",
    "UP2001_2024_01_23_Rehabilitation_day33",
    "UP2001_2024_01_25_Rehabilitation_day34",
    "UP2001_2024_01_29_Rehabilitation_day36",
    "UP2001_2024_01_30_Rehabilitation_day37",
    "UP2001_2024_02_01_Rehabilitation_day39",
    "UP2001_2024_02_02_Rehabilitation_day40",
    "UP2001_2024_02_05_Rehabilitation_day41",
    "UP2001_2024_02_06_Rehabilitation_day42",
    "UP2001_2024_02_07_Rehabilitation_day43",
    "UP2001_2024_02_08_Rehabilitation_day44",
    "UP2001_2024_02_09_Rehabilitation_day45",
    "UP2001_2024_02_13_Rehabilitation_day47",
    "UP2001_2024_02_14_Rehabilitation_day48",
    "UP2001_2024_02_15_Rehabilitation_day49",
    "UP2001_2024_02_16_Rehabilitation_day50",
    "UP2001_2024_02_21_PostRehab_ARAT_GRASSP_withBSI",
    "UP2001_2024_02_22_PostRehab_MVC",
    "UP2001_2024_02_23_MAS andRoM",
    "UP2001_2024_02_27_CUE-T",
    "UP2001_2024_02_29_SSEP",
    "UP2001_2024_03_19_ExtraSession_1",
    "UP2001_2024_04_02_ExtraSession_2",
    "UP2001_2024_04_26_ExtraSession_3",
    "UP2001_2024_04_30_ExtraSession_4",
    "UP2002_2024_05_14_ExtraSession_5",
]


def get_training_sessions(return_names: bool = False) -> list:
    """
    Return a list of training sessions.

    Parameters
    ----------
    return_names : bool, optional
        If True, return only the names of the training sessions.
        If False (default), return the full paths of the training sessions.

    Returns
    -------
    list
        A list of training sessions. Each session is either a full path or a name,
        depending on the value of the return_names parameter.
    """
    if return_names is True:
        return UP2001_TRAINING_SESSIONS
    assert os.environ["DATA_DIR"] != "", "DATA_DIR environment variable is not set"
    sessions_dir = Path(os.environ["DATA_DIR"]) / "__UP2" / "0_RAW_DATA" / "UP2_001"
    training_sessions = [s for s in list(Path(sessions_dir).glob("*")) if s.name in UP2001_TRAINING_SESSIONS]
    if len(training_sessions) != len(UP2001_TRAINING_SESSIONS):
        logger.warning(f"Only found {len(training_sessions)} out of {len(UP2001_TRAINING_SESSIONS)} training sessions")
    return training_sessions


def get_testing_sessions(return_names: bool = False) -> list:
    """
    Return a list of testing sessions.

    Parameters
    ----------
    return_names : bool, optional
        If True, return only the names of the testing sessions.
        If False (default), return the full paths of the testing sessions.

    Returns
    -------
    list
        A list of testing sessions. Each session is either a full path or a name,
        depending on the value of the return_names parameter.
    """
    if return_names is True:
        return UP2001_TESTING_SESSIONS
    assert os.environ["DATA_DIR"] != "", "DATA_DIR environment variable is not set"
    sessions_dir = Path(os.environ["DATA_DIR"]) / "__UP2" / "0_RAW_DATA" / "UP2_001"
    testing_sessions = [s for s in list(Path(sessions_dir).glob("*")) if s.name in UP2001_TESTING_SESSIONS]
    if len(testing_sessions) != len(UP2001_TESTING_SESSIONS):
        logger.warning(f"Only found {len(testing_sessions)} out of {len(UP2001_TESTING_SESSIONS)} testing sessions")
    return testing_sessions
