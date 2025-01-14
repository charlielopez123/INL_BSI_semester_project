""" Extract .smr brain signals into dictionary 

*Author: Kyuhwa Lee*
"""

import warnings

import neo.io
import numpy as np
from loguru import logger


def smr_to_dict(filename, debug=False):
    """
    Convert a .smr file (Spike2) to a dictionary.

    Parameters
    ----------
    filename : str
        The path to the .smr file.
    debug : bool, optional
        If True, enable debug logging. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the converted data from the .smr file. The dictionary has the following structure:
        ::

            {
                "STREAMS": {
                    "stream_name": {
                        "data": np.ndarray,
                        "t_start": float,
                        "t_stop": float,
                        "Fs": float,
                        "ch_names": list
                    },
                    ...
                }
            }

        - `stream_name` is the name of the stream in the .smr file.
        - `data` is a numpy array containing the signal data.
        - `t_start` is the start time of the signal.
        - `t_stop` is the stop time of the signal.
        - `Fs` is the sampling frequency of the signal.
        - `ch_names` is a list of channel names in the stream.

    Examples
    --------
    >>> smr_to_dict("/path/to/file.smr")

    """
    # aux function to read signal from .smr file (Spike2)
    with warnings.catch_warnings():  # avoid warning in Spike2IO
        warnings.simplefilter("ignore")
        reader = neo.io.Spike2IO(filename=filename)
        segment = reader.read_segment()

    # segment.analogsignals[0].array_annotations['channel_names'][0]
    nb_streams = len(
        segment.analogsignals
    )  # streams in the recording, could be power consumption streams, temperature, ..., we want the electrodes stream
    mdic = {"STREAMS": {}}

    for _stream_id in range(nb_streams):
        stream_name = segment.analogsignals[_stream_id].name
        if debug:
            logger.debug(stream_name)

        n_channels = segment.analogsignals[_stream_id].shape[
            1
        ]  # specific for BSI001 as segment.analogsignals[0] contains only nan, what is it for?
        fs = float(segment.analogsignals[_stream_id].sampling_rate)
        sigs = []

        t_start = float(segment.analogsignals[_stream_id].t_start)
        t_stop = float(segment.analogsignals[_stream_id].t_stop)
        n_samples_max = segment.analogsignals[_stream_id].shape[0]

        if debug:
            logger.debug("Number of channels: {:d}".format(n_channels))
            logger.debug("Number of samples: {:d}".format(n_samples_max))
            logger.debug("Sampling frequency: {:f}".format(fs))

        ch_names = list(segment.analogsignals[_stream_id].array_annotations["channel_names"])
        if debug:
            logger.debug("Channel names: {}".format(ch_names))

        for j in range(n_channels):
            channel = segment.analogsignals[_stream_id][:, j]
            sigs.append(np.array(channel[:n_samples_max]))

        sigs = np.asarray(sigs)

        no_spaces = stream_name.split()
        _name = ""
        for i in range(len(no_spaces) - 1):
            _name += no_spaces[i] + "_"
        _name += no_spaces[-1]

        mdic["STREAMS"][_name] = {
            "data": sigs,
            "t_start": t_start,
            "t_stop": t_stop,
            "Fs": fs,
            "ch_names": ch_names,
        }
    return mdic
