""" CHUV data parser

TODO: clarify the output of each scenario

Wimagine dt5 files have the following structure.

Top level: ['CommonData', 'ExperimentData', 'ScenarioData']

Level 'CommonData':
['ScenarioType']['SubScenarioNames']
['ScenarioType']['ScenarioName']

Level 'ScenarioData':
ndarray [0, 0]

Level 'ExperimentData':
list of HDF5 object references, where each object contains:
['AdditionalChannels', 'AlphaPredicted', 'AlphaWithSM', 'AlphaWithoutSM', 'EmiBias',
'EmiModel', 'ExpertTrained', 'IsFreestyleMode', 'IsUpdating', 'Latency', 'Prior',
'RawDataBuffer', 'ScenarioSupplementaryData', 'State', 'States', 'Time', 'Transition_Mat',
'Treatment', 'WeightSM', 'x', 'xLatent', 'y', 'yDesired']

'RawDataBuffer' of each object contains 59x32 chunks of floats.


*Author: Etienne de Montalivet, Kyuhwa Lee*
"""

import json
from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger

from lighthouse.data_loader.chuv_config import (
    ALPHA_JOINTS,
    CLASSIF_INDEPENDENT_SCENARIO,
    CLASSIF_SCENARIO,
)
from lighthouse.data_loader.load_smr import smr_to_dict
from lighthouse.utils import add_prefix, ascii_to_string, flatten_list

GDP_PRED_HEADER_SIZE = 5


def load_dt5(
    hdf_file,
    trigger_ch=None,
    return_ch_names: bool = False,
    return_y_desired: bool = False,
    return_y_pred: bool = False,
    return_states: bool = False,
    return_weight: bool = False,
    return_treatment: bool = False,
    return_pred_weight: bool = False,
    return_gamma: bool = False,
    return_alpha_pred: bool = False,
    return_latency: bool = False,
    return_alpha_with_sm: bool = False,
    return_alpha_without_sm: bool = False,
    return_time: bool = False,
    return_is_updating: bool = False,
    return_all: bool = False,
    verbose=False,
):
    """
    Load raw data blocks from a dt5 file and return raw signals including the trigger channel.

    Parameters
    ----------
    hdf_file : str
        The path to the dt5 file.
    trigger_ch : int or list, optional
        The trigger channel(s) to include. If None, all are returned with channel names
        `add_ch_<i>`. Defaults to None.
    return_ch_names : bool, optional
        Whether to return the channel names. Defaults to False.
    return_y_desired : bool, optional
        Whether to return the yDesired data. Defaults to False.
    return_y_pred : bool, optional
        Whether to return the yPred data. Defaults to False.
    return_states : bool, optional
        Whether to return the state data. Defaults to False.
    return_weight : bool, optional
        Whether to return the weight data. Defaults to False.
    return_treatment : bool, optional
        Whether to return the treatment data. Defaults to False.
    return_pred_weight : bool, optional
        Whether to return the prediction weight data. Defaults to False.
    return_gamma : bool, optional
        Whether to return the gamma data. Defaults to False.
    return_alpha_pred : bool, optional
        Whether to return the alpha predicted data. Defaults to False.
    return_latency : bool, optional
        Whether to return the latency data. Defaults to False.
    return_alpha_with_sm : bool, optional
        Whether to return the alpha with SM data. Defaults to False.
    return_alpha_without_sm : bool, optional
        Whether to return the alpha without SM data. Defaults to False.
    return_time : bool, optional
        Whether to return the time data. Defaults to False.
    return_is_updating : bool, optional
        Whether to return the is updating data. Defaults to False.
    return_all : bool, optional
        Whether to return all available data. Defaults to False.
    verbose : bool, optional
        Whether to print additional information. Defaults to False.

    Returns
    -------
    numpy.ndarray
        The raw signals in [channels x times] format.
    list (optional)
        The channel names. Format is: `<param>__<joint(opt)>__<action(opt)>`. If
        `<joint(opt)>__<action(opt)>` is unknown, then numbers corresponding to
        cells format in *hdf5* is used.
        In current version, ``param`` is one of the following:

            * states: unknown
            * state: cues value of every ``joint__action`` in CLASSIF_INDEPENDENT_SCENARIO
            * y_des: cues value of every ``joint__action`` in CLASSIF_SCENARIO
            * y_pred: predictions of Clinatec model
            * weight: unknown
            * treatment: unknown
            * pred_weight: weight of the prediction in the stimulation
            * gamma: unknown
            * alpha_pred: predictions of Clinatec model
            * latency: unknown. Probably the latency...
            * alpha_with_sm: unknown
            * alpha_without_sm: unknown
            * time: unknown

    Notes
    -----
    - The dt5 file lacks some headers such as sampling rate and channel names.
    - The trigger channel is included in the returned signals.

    ``AlphaPredicted`` if CLASSIF_INDEPENDENT_SCENARIO:
        * alpha_0: idle + 6 JOINTS
        * alpha_10: shoulder felexion extension idle
        * alpha_11: shoulder flexion
        * alpha_12: shoulder extension
        * alpha_20: shoulder adduction abduction idle
        * alpha_21: shoulder adduction
        * alpha_22: shoulder abduction
        * alpha_30: shoulder rotation idle
        * alpha_31: shoulder internal rotation
        * alpha_32: shoulder external rotation
        * alpha_40: elbow idle
        * alpha_41: elbow flexion
        * alpha_42: elbow extension
        * alpha_50: pronosup idle
        * alpha_51: pronation
        * alpha_52: supination
        * alpha_60: hand idle
        * alpha_61: hand open
        * alpha_62: hand close

    AlphaWithSM / AlphaWithoutSM:
        Somehow, it is related to AlphaPredicted hand (6). For example, it happened that
        ``alpha_with_sm_<i> == alpha_pred_6<i>`` for all i. Same for ``alpha_without_sm_<i>``.

    Raises
    ------
    Exception
        If there is an error getting metadata from the chuv files.

    Examples
    --------
    >>> import os
    >>> from pathlib import Path
    >>> from lighthouse.data_loader.load_chuv import load_dt5

    >>> data_dir = Path(os.environ["DATA_DIR"]) / "__UP2" / "0_RAW_DATA" / "UP2_001"
    >>> dt5_file = (
    ...     data_dir
    ...     / "UP2001_2023_11_02_BSITraining_day11"
    ...     / "WISCI"
    ...     / "2023_11_02_13_23_37 (rec1)"
    ...     / "ABSD data"
    ...     / "2023_11_02_13_27_38_01of02.dt5"
    ... )
    >>> signals, ch_names = load_dt5(dt5_file, return_ch_names=True, return_all=True)
    """
    if trigger_ch is None:
        trigger_ch = "all"
    if isinstance(trigger_ch, int):
        trigger_ch = [trigger_ch]
    hdf = h5py.File(hdf_file, "r", driver="core")
    if "CommonData" in hdf.keys():
        try:
            logger.info(
                f"Scenario Name: {ascii_to_string(hdf['CommonData']['ScenarioType']['ScenarioName'][()].flatten())}"
            )
            logger.info(
                f"Sub Scenario Name: {ascii_to_string(hdf[hdf['CommonData']['ScenarioType']['SubScenarioNames'][()][0][0]][()].flatten())}"
            )
        except Exception as e:
            logger.warning(f"Cannot get metadata from chuv files: {e}")
    data_yDesired = []
    data_y = []
    data_states = []
    data_state = []
    data_blocks = []
    data_triggers = []
    data_weight = []
    data_treatment = []
    data_pred_weight = []
    data_gamma = []
    data_alpha_pred = []
    alpha_pred_ch_names = None
    data_latency = []
    data_alpha_with_sm = []
    data_alpha_without_sm = []
    data_time = []
    data_is_updating = []
    n_blocks = hdf["ExperimentData"].shape[-1]
    for i_block in range(n_blocks):
        ref = hdf["ExperimentData"][0][i_block]
        data_blocks.append(hdf[ref]["RawDataBuffer"][()])
        data_triggers.append(hdf[ref]["AdditionalChannels"][()])
        if return_y_desired or return_all:
            data_yDesired.append(hdf[ref]["yDesired"][()])
        if return_y_pred or return_all:
            data_y.append(hdf[ref]["y"][()])
        if return_states or return_all:
            data_states.append(hdf[ref]["States"][()])
            data_state.append(hdf[ref]["State"][()])
        if return_weight or return_all:
            data_weight.append(hdf[ref]["WeightSM"][()][0][0])
        if return_treatment or return_all:
            data_treatment.append(hdf[ref]["Treatment"][()][0][0])
        if return_pred_weight or return_all:
            try:
                data_pred_weight.append(hdf[ref]["ScenarioSupplementaryData"]["PredictionWeight"][()][0][0])
            except Exception as _:
                try:
                    new_ref = hdf[ref]["ScenarioSupplementaryData"][()][0][0]
                    data_pred_weight.append(hdf[new_ref]["PredictionWeight"][()][0][0])
                except Exception as _:
                    data_pred_weight.append(np.nan)
        if return_gamma or return_all:
            try:
                data_gamma.append(hdf[ref]["GammaPredicted"][()])
            except Exception as e:
                data_gamma.append(np.nan)
        if return_latency or return_all:
            data_latency.append(hdf[ref]["Latency"][()])
        if return_alpha_pred or return_all:
            data_alpha_pred.append(
                np.hstack(
                    [hdf[hdf[ref]["AlphaPredicted"][()][i][0]][()] for i in range(len(hdf[ref]["AlphaPredicted"][()]))]
                )
            )
            if alpha_pred_ch_names is None:  # TODO fix
                alpha_pred_ch_names = np.hstack(
                    [
                        [
                            f"alpha_{i_ref}{j}"
                            for j in range((hdf[hdf[ref]["AlphaPredicted"][()][i_ref][0]][()]).shape[1])
                        ]
                        for i_ref in range(len(hdf[ref]["AlphaPredicted"][()]))
                    ]
                )
        if return_alpha_with_sm or return_all:
            data_alpha_with_sm.append(hdf[ref]["AlphaWithSM"][()])
        if return_alpha_without_sm or return_all:
            data_alpha_without_sm.append(hdf[ref]["AlphaWithoutSM"][()])
        if return_time or return_all:
            data_time.append(hdf[ref]["Time"][()])
        if return_is_updating or return_all:
            data_is_updating.append(hdf[ref]["IsUpdating"][()])
    ##############################################
    # put all signals in [channels x times] format
    signals = np.concatenate(data_blocks, axis=0).T  # [channels x times]
    data_triggers = np.concatenate(data_triggers, axis=0).T  # [channels x times]
    if return_y_desired or return_all:
        data_yDesired = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_yDesired,
            )
            .squeeze()
            .T
        )  # [channels x times]
    if return_y_pred or return_all:
        data_y = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_y,
            )
            .squeeze()
            .T
        )  # [channels x times]
    if return_states or return_all:
        data_states = np.repeat(data_states, 59)[None, :]
        data_state = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_state,
            )
            .squeeze()
            .T
        )
    if return_weight or return_all:
        data_weight = np.repeat(data_weight, 59)[None, :]
    if return_treatment or return_all:
        data_treatment = np.repeat(data_treatment, 59)[None, :]
    if return_pred_weight or return_all:
        data_pred_weight = np.repeat(data_pred_weight, 59)[None, :]
    if return_gamma or return_all:
        data_gamma = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_gamma,
            )
            .squeeze()
            .T
        )  # [channels x times]
    if return_alpha_pred or return_all:
        data_alpha_pred = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_alpha_pred,
            )
            .squeeze()
            .T
        )  # [channels x times]
    if return_latency or return_all:
        data_latency = np.repeat(data_latency, 59)[None, :]
    if return_alpha_without_sm or return_all:
        data_alpha_without_sm = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_alpha_without_sm,
            )
            .squeeze()
            .T
        )  # [channels x times]
    if return_alpha_with_sm or return_all:
        data_alpha_with_sm = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_alpha_with_sm,
            )
            .squeeze()
            .T
        )  # [channels x times]
    if return_time or return_all:
        data_time = (
            np.apply_along_axis(
                lambda x: np.repeat(x, 59),
                0,
                data_time,
            )
            .squeeze()
            .T
        )
    if return_is_updating or return_all:
        data_is_updating = np.repeat(data_is_updating, 59)[None, :]

    # extract trigger channel
    data_triggers = np.nan_to_num(data_triggers, nan=0.0)  # convert nan to 0
    if trigger_ch == "all":
        trigger_ch = range(data_triggers.shape[0])
    ##############################################
    # stack all channels + CHANNEL NAMES
    ch_names = [f"ecog_{i}" for i in range(signals.shape[0])]
    if return_states or return_all:
        signals = np.vstack([data_state, signals])
        signals = np.vstack([data_states, signals])
        if data_state.shape[0] == len(CLASSIF_INDEPENDENT_SCENARIO):
            ch_names = ["states"] + add_prefix("state__", CLASSIF_INDEPENDENT_SCENARIO) + ch_names
        elif data_state.shape[0] == len(CLASSIF_SCENARIO):
            ch_names = ["states"] + add_prefix("state__", CLASSIF_SCENARIO) + ch_names
        else:
            logger.warning("Unknown scenario!")
            ch_names = ["states"] + [f"state_{i}" for i in range(data_state.shape[0])] + ch_names
    if return_y_desired or return_all:
        signals = np.vstack([data_yDesired, signals])  # add additional channels
        if data_yDesired.shape[0] == len(CLASSIF_INDEPENDENT_SCENARIO[1:]):
            ch_names = add_prefix("y_des__", CLASSIF_INDEPENDENT_SCENARIO[1:]) + ch_names
        elif data_yDesired.shape[0] == len(CLASSIF_SCENARIO[1:]):
            ch_names = add_prefix("y_des__", CLASSIF_SCENARIO[1:]) + ch_names
        else:
            logger.warning("Unknown scenario!")
            ch_names = [f"y_des__{i}" for i in range(data_yDesired.shape[0])] + ch_names
    if return_y_pred or return_all:
        signals = np.vstack([data_y, signals])
        if data_y.shape[0] == len(CLASSIF_INDEPENDENT_SCENARIO[1:]):
            ch_names = add_prefix("y_pred__", CLASSIF_INDEPENDENT_SCENARIO[1:]) + ch_names
        elif data_y.shape[0] == len(CLASSIF_SCENARIO[1:]):
            ch_names = add_prefix("y_pred__", CLASSIF_SCENARIO[1:]) + ch_names
        else:
            logger.warning("Unknown scenario!")
            ch_names = [f"y_pred__{i}" for i in range(data_y.shape[0])] + ch_names
    if return_weight or return_all:
        signals = np.vstack([data_weight, signals])
        ch_names = ["weight"] + ch_names
    if return_treatment or return_all:
        signals = np.vstack([data_treatment, signals])
        ch_names = ["treatment"] + ch_names
    if return_pred_weight or return_all:
        if np.isnan(data_pred_weight).all():
            logger.warning("Skipping pred_weight as it is all nan")
        else:
            signals = np.vstack([data_pred_weight, signals])
            ch_names = ["pred_weight"] + ch_names
    if return_gamma or return_all:
        if np.isnan(data_gamma).all():
            logger.warning("Skipping gamma as it is all nan")
        else:
            signals = np.vstack([data_gamma, signals])
            ch_names = [f"gamma__{i}" for i in range(data_gamma.shape[0])] + ch_names
    if return_alpha_pred or return_all:
        signals = np.vstack([data_alpha_pred, signals])
        if data_alpha_pred.shape[0] == len(CLASSIF_INDEPENDENT_SCENARIO[1:]):
            ch_names = (
                [f"alpha_pred__{joint}" for joint in ALPHA_JOINTS]
                + add_prefix("alpha_pred__", CLASSIF_INDEPENDENT_SCENARIO[1:])
                + ch_names
            )
        elif data_alpha_pred.shape[0] == len(CLASSIF_SCENARIO):
            ch_names = add_prefix("alpha_pred__", CLASSIF_SCENARIO) + ch_names
        else:
            logger.warning("Unknown scenario!")
            ch_names = list(alpha_pred_ch_names) + ch_names
    if return_latency or return_all:
        signals = np.vstack([data_latency, signals])
        ch_names = ["latency"] + ch_names
    if return_alpha_with_sm or return_all:
        signals = np.vstack([data_alpha_with_sm, signals])
        ch_names = [f"alpha_with_sm__{i}" for i in range(data_alpha_with_sm.shape[0])] + ch_names
    if return_alpha_without_sm or return_all:
        signals = np.vstack([data_alpha_without_sm, signals])
        ch_names = [f"alpha_without_sm__{i}" for i in range(data_alpha_without_sm.shape[0])] + ch_names
    if return_time or return_all:
        signals = np.vstack([data_time, signals])
        ch_names = [f"time__{i}" for i in range(data_time.shape[0])] + ch_names
    if return_is_updating or return_all:
        signals = np.vstack([data_is_updating, signals])
        ch_names = ["is_updating"] + ch_names
    # add trigger channels at the end
    signals = np.vstack([data_triggers[trigger_ch, :], signals])  # add trigger channel
    ch_names = [f"add_ch_{i}" for i in range(len(trigger_ch))] + ch_names
    if verbose:
        for i, row in enumerate(data_triggers):
            logger.debug("add_ch%d: %s" % (i, np.unique(row)))
    if return_ch_names:
        assert len(signals) == len(ch_names), "Number of channels and channel names do not match. Incorrect scenario?"
        return signals, ch_names

    return signals


def concatenate_dt5(dt5_files, trigger_ch=None, return_ch_names: bool = False, verbose=False, **kwargs):
    """
    Concatenates signals from successive dt5 files along the time axis.

    Parameters
    ----------
    dt5_files : list
        List of dt5 file paths.
    trigger_ch : str, optional
        Name of the trigger channel, by default None.
    return_ch_names : bool, optional
        Whether to return channel names along with concatenated signals, by default False.
    verbose : bool, optional
        Whether to print loading progress, by default False.
    **kwargs : dict
        Additional keyword arguments to be passed to the `load_dt5` function.

    Notes
    -----
    dt5 files must be contiguous in time. The function only concatenates the signals along
    the time axis.

    Returns
    -------
    np.ndarray or tuple
        If `return_ch_names` is False, returns a numpy array containing the concatenated signals.
        If `return_ch_names` is True, returns a tuple containing the concatenated signals and the channel names.
    """
    signals = []
    for dt5_file in dt5_files:
        logger.info(f"Loading {dt5_file}")
        if return_ch_names:
            signals_file, ch_names = load_dt5(
                dt5_file,
                trigger_ch,
                return_ch_names,
                verbose,
                **kwargs,
            )
        else:
            signals_file = load_dt5(
                dt5_file,
                trigger_ch,
                return_ch_names,
                verbose,
                **kwargs,
            )
        signals.append(signals_file)
    # concatenate signals along the time axis
    if return_ch_names:
        return np.concatenate(signals, axis=1), ch_names
    return np.concatenate(signals, axis=1)


def load_absd(absd_dir: Union[Path, str], **kwargs) -> tuple:
    """Load ABSD data from the given directory

    Parameters
    ----------
    absd_dir : Union[Path, str]
        The directory containing all dt5 files
    **kwargs: dict
        Additional keyword arguments to be passed to the `concatenate_dt5` function.
        These arguments are then passed to the `load_dt5` function. See notes.

    Returns
    -------
    tuple
        - np.ndarray: absd data in format `(n_channels, n_samples)`
        - list: absd channel names

    Notes
    -----
    Don't use `return_ch_names` in the `**kwargs` as it's already `True`
    by default and this is fixed behavior. See `load_dt5` params for more details.

    Examples
    --------
    >>> import os
    >>> from pathlib import Path
    >>> from lighthouse.data_loader.load_chuv import load_absd

    >>> data_dir = Path(os.environ["DATA_DIR"]) / "__UP2" / "0_RAW_DATA" / "UP2_001"
    >>> absd_dir = (
    ...     data_dir
    ...     / "UP2001_2023_11_02_BSITraining_day11"
    ...     / "WISCI"
    ...     / "2023_11_02_13_23_37 (rec1)"
    ...     / "ABSD data"
    ... )
    >>> signals, ch_names = load_absd(absd_dir)
    """
    dt5_files = sorted(list(Path(absd_dir).glob("*.dt5")))
    # check list
    if len(dt5_files) == 0:
        raise ValueError(f"No dt5 files found in {absd_dir}")
    n_files = len(dt5_files)
    counter = 1
    for file in dt5_files:
        if not int(file.stem[-2:]) == n_files:
            raise ValueError(f"Unexpected value in filename: expected {n_files} and got {file.stem[-2:]}")
        if not counter == int(file.stem[-6:-4]):
            raise ValueError(f"Order of files is not correct: expected {counter} and got {int(file.stem[-6:-4])}")
        counter += 1
    # load data
    absd_data, absd_ch_names = concatenate_dt5(
        dt5_files,
        return_ch_names=True,
        **kwargs,
    )
    return absd_data, absd_ch_names


def load_gdp(gdp_dir: Union[Path, str], use_external_time_ref: bool = True, debug=False):
    """
    Load stimulation data from a directory.

    Parameters
    ----------
    gdp_dir : str
        The directory path containing the gdp files.
    use_external_time_ref : str, optional
        If True, time reference used is from ``ExternalTime`` column (so the Unix time). Else,
        ``GDPTime`` column is used for time reference. Default is True.
    debug : bool, optional
        Whether to print debug information. Default is False.

    Returns
    -------
    tuple
        A tuple containing the following elements:

        - Tuple: A tuple containing the stimulation amplitudes `(n_channels, n_times)`,
          stimulation times `(n_times,)`, and stimulation channel names. If stim data is not
          consistent across patterns, the amplitudes and times are returned as a list of arrays.
        - Tuple: A tuple containing the prediction data `(n_channels, n_times)`, prediction
          times `(n_times,)`, and prediction channel names.
        - Tuple: A tuple containing the enable stimulation values `(n_events,)` and enable
          stimulation times `(n_ev_times,)`.
        - numpy.ndarray: An array containing the loop modes.
        - dict: The stimulation metadata per stim event. Keys are the stimulation pattern names,
          values are the corresponding metadata pandas dataframe format.

    Raises
    ------
    ValueError
        If the data is not consistent across predictions.

    Examples
    --------
    >>> import os
    >>> from pathlib import Path
    >>> from lighthouse.data_loader.load_chuv import load_gdp

    >>> data_dir = Path(os.environ["DATA_DIR"]) / "__UP2" / "0_RAW_DATA" / "UP2_001"
    >>> (
    ...     (stim_data, stim_times, stim_ch_names),
    ...     (pred_data, pred_times, pred_ch_names),
    ...     (enable_stim_data, enable_stim_times),
    ...     lm,
    ...     stim_metadata,
    ... ) = load_gdp(
    ...     data_dir
    ...     / "UP2001_2023_11_02_BSITraining_day11"
    ...     / "GDP"
    ...     / "Patients"
    ...     / "Patient_UP2001Rostral"
    ...     / "Sessions"
    ...     / "Session_20231102141829"
    ...     / "GeneralLogs"
    ... )

    """
    if not isinstance(gdp_dir, Path):
        gdp_dir = Path(gdp_dir)
    assert gdp_dir.exists(), f"{gdp_dir} does not exist"

    stim_columns = [
        "Electrodes",
        "LeadNum",
        "FrequencyPeriod",
        "FrequencyOffset",
        "Amplitude",
        "RelativeAmplitude",
        "Ramping",
        "Pulses",
        "ModulationPeriod",
        "Duration",
        "RampingDuration",
        "gdp_time",
    ]

    gdp_times = None
    external_times = None
    linreg = None
    pred_times = None
    pred_data = []
    stim_data = []
    stim_times = []
    stim_metadatas = {}
    enable_stim_times = []
    enable_stim_data = []
    pred_ch_names = []
    loop_modes = []

    # check prediction consistency (number of predictions in files)
    min_pred_len = np.inf
    for file in Path(gdp_dir).glob("*"):
        if "prediction" in file.name:
            # prediction consistency
            # check if file not empty of predictions
            with open(file, encoding="utf8") as f:
                n_lines = sum(1 for _ in f)
            if n_lines == GDP_PRED_HEADER_SIZE:
                raise ValueError(
                    "prediction file with no prediction is not supported yet. Either update "
                    + "the code or delete this file."
                )
            df = pd.read_csv(file, sep=";", header=GDP_PRED_HEADER_SIZE, usecols=[0, 1, 2])
            if debug:
                logger.debug(f"{file.name} has {len(df['GDPTime'])} predictions")
            if len(df["GDPTime"]) < min_pred_len:
                min_pred_len = len(df["GDPTime"])
    # linear fit between GDPTime and ExternalTime
    for file in Path(gdp_dir).glob("*"):
        if "prediction" in file.name:
            gdp_times = df["GDPTime"]
            external_times = df["ExternalTime"]
            linreg = scipy.stats.linregress(gdp_times, external_times)
            break
    # check if multiple prediction files - not supported yet
    # TODO: support loading gdp dir with multiple predictions
    time_suffixes = []
    for file in Path(gdp_dir).glob("*"):
        if "prediction" in file.name:
            time_suffixes.append(str(file.stem).split("_")[-1])
    if len(np.unique(time_suffixes)) > 1:
        raise ValueError(
            f"More than one set of predictions is not supported yet. Got "
            + f"{len(np.unique(time_suffixes))}: {list(np.unique(time_suffixes))}"
        )
    #
    is_session_empty = True
    for file in Path(gdp_dir).glob("*"):
        if "prediction" in file.name:
            if min_pred_len == 0:
                continue
            df = pd.read_csv(
                file,
                sep=";",
                header=GDP_PRED_HEADER_SIZE,
                usecols=[0, 1, 2],
                nrows=min_pred_len,
            )
            pred_data.append(df["Value"].values)
            pred_ch_names.append(file.name.split("prediction_")[0])
            if pred_times is None:
                pred_times = df["GDPTime"].values
            else:
                if (df["GDPTime"].values != pred_times).all():
                    raise ValueError("Data is not consistent across predictions")
        elif "OWDActivity" in file.name:
            with open(file, "r", encoding="utf8") as f:
                header_over = False
                for i, line in enumerate(f):
                    if header_over is False:
                        if "============" in line:
                            header_over = True
                    else:
                        is_session_empty = False
                        if "Success Stim" in line:
                            if "is_enable" not in stim_metadatas:
                                stim_metadatas["is_enable"] = []
                            enable_stim_times.append(int(line.split("\t")[0]))
                            enable_stim_data.append(1 if "StimOn" in line else 0)
                            stim_metadatas["is_enable"].append([int(line.split("\t")[0]), 1 if "StimOn" in line else 0])
                        elif "Success" in line:
                            json_data = json.loads("{" + line.split("\t")[1].split("{", 1)[1])
                            if json_data["LoopMode"] > 0:
                                raise ValueError("loopmode > 0 are not implemented yet.")
                            for i in range(len(json_data["Waveforms"])):
                                # add key if not exists
                                if json_data["Waveforms"][i]["Name"] not in stim_metadatas:
                                    stim_metadatas[json_data["Waveforms"][i]["Name"]] = []
                                # append all metadata
                                stim_metadatas[json_data["Waveforms"][i]["Name"]].append(
                                    flatten_list(
                                        [
                                            [v for k, v in json_data["Waveforms"][i].items() if k != "Name"],
                                            [
                                                v
                                                for k, v in json_data["StimColumns"][0]["StimRows"][i].items()
                                                if k != "WaveformIndex"
                                            ],
                                            [v for k, v in json_data["StimColumns"][0].items() if k != "StimRows"],
                                            [
                                                int(line.split("\t")[0]),
                                            ],
                                        ]
                                    )
                                )
                            loop_modes.append(json_data["LoopMode"])
                        else:
                            if debug:
                                logger.debug(f"Unknown line: \n{line}")
        else:
            if debug:
                logger.debug(f"Unparsed file: {file.name} - {file}")
    # if empty session, return empty arrays
    if is_session_empty is True:
        return (
            (
                np.array([]),
                np.array([]),
                [],
            ),
            (
                np.array([]),
                np.array([]),
                [],
            ),
            (
                np.array([]),
                [],
            ),
            np.array([]),
            {},
        )

    stim_metadatas = {
        k: pd.DataFrame(
            stim_metadatas[k],
            columns=stim_columns if k != "is_enable" else ["gdp_time", "is_enable"],
        )
        for k in stim_metadatas
    }
    stim_ch_names = [p for p in list(stim_metadatas) if p != "is_enable"]

    ### get stim=amplitude data
    try:
        stim_data = np.vstack([df["Amplitude"].values for k, df in stim_metadatas.items() if k != "is_enable"])
        ### common stim times
        # get the first pattern name different from is_enable
        try:
            first_pattern = (
                list(stim_metadatas)[0] if list(stim_metadatas)[0] != "is_enable" else list(stim_metadatas)[1]
            )
        except Exception as e:
            raise ValueError(f"Session with only general stim on/off without actual stim is not supported yet.")
        stim_times = stim_metadatas[first_pattern]["gdp_time"].values

    except ValueError as e:
        logger.warning(
            "For unknow reason, the amplitude is not set for all stim. Returning list"
            + " of stim data instead of array"
        )
        stim_data = [df["Amplitude"].values for k, df in stim_metadatas.items() if k != "is_enable"]
        stim_times = [stim_metadatas[k]["gdp_time"].values for k in list(stim_metadatas) if k != "is_enable"]

    pred_data = np.array(pred_data)
    # force start/end of enable stim to be 0 if predictions are available
    if pred_times is not None:
        # TODO: double check with Henri's team that gdp is always off when it starts
        if len(enable_stim_data) == 0 or enable_stim_data[0] != 0:
            enable_stim_data = [0] + enable_stim_data
            enable_stim_times = [pred_times[0]] + enable_stim_times
        if enable_stim_data[-1] != 0:
            enable_stim_data = enable_stim_data + [0]
            enable_stim_times = enable_stim_times + [pred_times[-1]]
    enable_stim_data = np.array(enable_stim_data)
    enable_stim_times = np.array(enable_stim_times)
    # use GDPTime or ExternalTime reference
    if use_external_time_ref is True:
        if linreg is None:
            logger.warning("Cannot use ExternalTime reference. Maybe no prediction files ?")
        else:
            if isinstance(stim_times, list):
                stim_times = [stim_time * linreg.slope + linreg.intercept for stim_time in stim_times]
            else:
                stim_times = stim_times * linreg.slope + linreg.intercept
            if pred_times is not None:
                pred_times = pred_times * linreg.slope + linreg.intercept
            enable_stim_times = enable_stim_times * linreg.slope + linreg.intercept
    # anyway add external_time column in metadata if possible
    if linreg is not None:
        for joint in stim_metadatas:
            stim_metadatas[joint]["external_time"] = stim_metadatas[joint]["gdp_time"] * linreg.slope + linreg.intercept
    # return empty list instead of None
    if pred_times is None:
        pred_times = np.array([])
    return (
        (stim_data, stim_times, stim_ch_names),
        (pred_data, pred_times, pred_ch_names),
        (enable_stim_data, enable_stim_times),
        np.array(loop_modes),
        stim_metadatas,
    )


def concat_gdp(gdp_dirs: List[Union[Path, str]], debug: bool) -> tuple:
    """Concat the data from multiple GDP recordings.

    Parameters
    ----------
    gdp_dirs : List[Union[Path, str]]
        List of paths to GDP directories
    debug : bool
        Verbose mode.

    Returns
    -------
    A tuple containing the following elements:
        - Tuple: A tuple containing the stimulation amplitudes `(n_channels, n_times)`,
            stimulation times `(n_times,)`, and stimulation channel names.
        - Tuple: A tuple containing the prediction data `(n_channels, n_times)`, prediction
            times `(n_times,)`, and prediction channel names.
        - Tuple: A tuple containing the enable stimulation values `(n_events,)` and enable
            stimulation times `(n_ev_times,)`.
        - numpy.ndarray: An array containing the loop modes.
        - dict: The stimulation metadata per stim event. Keys are the stimulation pattern names,
            values are the corresponding metadata pandas dataframe format.

    Notes
    -----
    Because different stim patterns could have been used across the recordings, the returned
    stimulation amplitudes are filled with 0s when a given pattern was not used. Exact information
    (`0` vs `no data`) is to be found in `stim_metadata`
    """
    all_stim_amplitudes = []
    all_stim_times = []
    all_stim_ch_names = []
    all_pred_data = []
    all_pred_times = []
    all_pred_ch_names = None
    all_enable_stim_values = []
    all_enable_stim_times = []
    all_loop_modes = []
    all_stim_metadata = []

    for gdp_dir in gdp_dirs:
        logger.info(f"Loading data from {gdp_dir}")
        (
            (stim_amplitudes, stim_times, stim_ch_names),
            (pred_data, pred_times, pred_ch_names),
            (enable_stim_values, enable_stim_times),
            loop_modes,
            stim_metadata,
        ) = load_gdp(gdp_dir, use_external_time_ref=True, debug=debug)
        # stim
        all_stim_amplitudes.append(stim_amplitudes)
        all_stim_times.append(stim_times)
        all_stim_ch_names.append(stim_ch_names)

        # pred
        all_pred_data.append(pred_data)
        all_pred_times.append(pred_times)
        if all_pred_ch_names is None:
            all_pred_ch_names = pred_ch_names
        else:
            assert all_pred_ch_names == pred_ch_names
        # enable stim
        all_enable_stim_values.append(enable_stim_values)
        all_enable_stim_times.append(enable_stim_times)
        # loop modes
        all_loop_modes.append(loop_modes)
        all_stim_metadata.append(stim_metadata)

    ### concat stim
    # concatenate data across recordings per channel
    concat_stim_ch_names = list(np.unique(flatten_list(all_stim_ch_names)))
    concat_stim_data = []
    for ch in concat_stim_ch_names:
        concat_stim_data_per_ch = []
        for stim_ch_names, stim_data in zip(all_stim_ch_names, all_stim_amplitudes):
            if ch in stim_ch_names:
                concat_stim_data_per_ch.append(stim_data[stim_ch_names.index(ch)])
            else:
                concat_stim_data_per_ch.append(np.zeros(stim_data.shape[1]))
        concat_stim_data.append(np.concatenate(concat_stim_data_per_ch))
    concat_stim_data = np.array(concat_stim_data)
    concat_stim_times = np.concatenate(all_stim_times)

    ### concat metadata
    concat_stim_metadata = {}
    for joint in list(np.unique(flatten_list(all_stim_ch_names))):
        metadata_per_ch = []
        for stim_metadata in all_stim_metadata:
            if joint in stim_metadata:
                metadata_per_ch.append(stim_metadata[joint])
        concat_stim_metadata[joint] = pd.concat(metadata_per_ch)

    ### concat pred data
    concat_pred_data = np.hstack(all_pred_data)
    concat_pred_ch_names = all_pred_ch_names
    concat_pred_times = np.concatenate(all_pred_times)

    ### concat enable stim
    concat_enable_stim_data = np.concatenate(all_enable_stim_values)
    concat_enable_stim_times = np.concatenate(all_enable_stim_times)

    ### concat loop modes
    concat_loop_modes = np.concatenate(all_loop_modes)

    return (
        (concat_stim_data, concat_stim_times, concat_stim_ch_names),
        (concat_pred_data, concat_pred_times, concat_pred_ch_names),
        (concat_enable_stim_data, concat_enable_stim_times),
        concat_loop_modes,
        concat_stim_metadata,
    )


def load_smr(smr_file, trigger_name: str = None, rectify_trigger: bool = True, debug=False):
    """
    Load SMR file and return the signals, time array, and channel names.

    Parameters
    ----------
    smr_file : str
        The path to the SMR file to be loaded.
    trigger_name : str, optional
        The name of the trigger channel. If None, no rectification is applied. The default is None.
    rectify_trigger : bool, optional
        Whether to rectify the trigger channel, by default True.
    debug : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    tuple
        A tuple containing the following elements:

        - signals : numpy.ndarray
            The loaded signals from the SMR file.
        - time_array : numpy.ndarray
            The time array corresponding to the loaded signals.
        - ch_names : list
            The names of the channels in the SMR file.

    Raises
    ------
    AssertionError
        If the sampling frequency, start time, or stop time of the streams in the SMR file are
        inconsistent.

    Notes
    -----
    This function assumes that the SMR file contains multiple streams, each representing a
    different channel. The function extracts the signals, time array, and channel names from
    the SMR file. If a trigger channel is given and `rectify_trigger` is True the trigger channel
    will be rectified.

    No times is associated with the signals. The time array is generated from the sampling
    frequency and always starts at 0.

    Examples
    --------
    >>> import os
    >>> from pathlib import Path
    >>> from lighthouse.data_loader.load_chuv import load_smr

    >>> data_dir = Path(os.environ["DATA_DIR"]) / "__UP2" / "0_RAW_DATA" / "UP2_001"
    >>> smr_file = (
    ...     data_dir
    ...     / "UP2001_2023_11_02_BSITraining_day11"
    ...     / "WISCI"
    ...     / "2023_11_02_13_23_37 (rec1)"
    ...     / "SN07_merged.smr"
    ... )
    >>> signals, times, ch_names = load_smr(smr_file)

    """
    logger.info(f"Loading {smr_file}")
    mdic = smr_to_dict(smr_file)

    sfreq = None
    t_start = None
    t_stop = None
    ch_names = []
    signals = []
    ch_types = []
    for name, stream in mdic["STREAMS"].items():
        if sfreq is None:
            sfreq = stream["Fs"]
        else:
            assert sfreq == stream["Fs"], "Inconsistent sampling frequency"

        if t_start is None:
            t_start = stream["t_start"]
            if debug:
                logger.debug(t_start)
        else:
            assert t_start == stream["t_start"], "Inconsistent start time"

        if t_stop is None:
            t_stop = stream["t_stop"]
        else:
            assert t_stop == stream["t_stop"], "Inconsistent stop time"

        ch_names.extend(stream["ch_names"])
        assert stream["data"].shape[2] == 1  # unnecessary extra dimension
        signals.append(stream["data"][:, :, 0])
        if name == "Signal_stream_3":
            if stream["data"].shape[0] != 32:
                logger.warning("Signal_stream_3 does not have 32 channels. Double check.")
            ch_types.extend(["ecog"] * len(stream["data"]))
        else:
            ch_types.extend(["misc"] * len(stream["data"]))

    signals = np.concatenate(signals, axis=0)

    if trigger_name is not None and rectify_trigger is True:
        trigger_ch = np.where(np.array(ch_names) == trigger_name)[0][0]
        pwm_length = 200  # in ms
        pwm_window = int(sfreq * (pwm_length / 1000))

        tr7 = np.nan_to_num(signals[trigger_ch], nan=0)
        tr7_rectified = tr7.copy()
        i = 0
        while i < (len(tr7) - pwm_window):
            peak_start = np.where(tr7[i : i + pwm_window] > 0)[0]
            if len(peak_start) > 0:
                # ON state
                j = i + peak_start[0] + 1
                while True:
                    next_peak = np.where(tr7[j : j + pwm_window] > 0)[0]
                    if len(next_peak) == 0:
                        break
                    j += next_peak[-1] + 1
                tr7_rectified[i:j] = 1
                if debug:
                    logger.debug(f"LED ON {i} -> {j}")
                i = j + 1
            else:
                # OFF state
                i += pwm_window
        signals[trigger_ch] = tr7_rectified

    return signals, np.arange(t_start, t_stop, 1 / sfreq), ch_names
