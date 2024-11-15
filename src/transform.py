""" This module contains classes and functions to transform data in torch-like datasets.

In particular, the following transformations are defined:

* MNEFilter: Filter data using MNE functions.

    
*Author: Etienne de Montalivet*
"""

import mne
import numpy as np


class MNEFilter(object):
    """Class to filter data using MNE functions to be used in a transform pipeline.

    Parameters
    ----------
    sfreq : int
        Sampling frequency.
    l_freq : int
        Low frequency for bandpass.
    h_freq : int
        High frequency for bandpass.
    notch_freqs : list, optional
        List of frequencies to notch filter. Default is [50, 100, 150, 200].
    apply_car : bool, optional
        Whether to apply the common average reference. Default is False.

    Examples
    --------
    >>> import torchvision.transforms as T
    >>> from syn_decoder.transform import MNEFilter
    >>> transform = T.Compose(
    ...     [
    ...        MNEFilter(sfreq=SFREQ, l_freq=1, h_freq=200, notch_freqs=[50, 100, 150, 200], apply_car=True),
    ...        torchvision.transforms.ToTensor(),
    ...     ],
    ... )
    """

    def __init__(
        self, sfreq: int, l_freq: int, h_freq: int, notch_freqs: list = [50, 100, 150, 200], apply_car: bool = False
    ):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = notch_freqs
        self.apply_car = apply_car

    def __call__(self, data: np.ndarray):
        """data has shape (n_channels, n_samples)"""
        if len(data.shape) != 2:
            raise ValueError("Data must be a 2D array")
        ch_names = [f"ch_{i}" for i in range(data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="ecog")
        raw = mne.io.RawArray(data, info)
        # CAR
        if self.apply_car:
            raw.set_eeg_reference(ref_channels="average", projection=False)
        # NOTCH
        if self.notch_freqs is not None:
            raw.notch_filter(freqs=self.notch_freqs, notch_widths=2, fir_design="firwin")
        # BANDPASS
        if self.h_freq is not None and self.l_freq is not None:
            raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design="firwin")

        return raw.get_data()
