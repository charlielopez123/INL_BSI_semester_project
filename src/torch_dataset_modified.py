"""
This module contains a torch dataset to load time series data.

Authors: Etienne de Montalivet
"""

from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class TimeseriesDataset(Dataset):
    """A torch dataset for time series data.

    This dataset can be used for any type of time series data. It takes a time series loader
    function and its parameters to load the data. The data is then segmented into windows
    of size `n_samples_win` with a stride of `n_samples_step`. The data can be preprocessed
    using `x_preprocess`. The windowed data can be then transformed using `x_transform`.

    All this pipeline can be applied to the target/label data as well if `load_y_func` is provided.

    Parameters
    ----------
    load_x_func : callable
        Function to load the input data.
    load_x_args : dict
        Parameters to pass to `load_x_func`. If using a function that takes no parameters (such as lambda function),
        pass None.
    n_samples_step : int
        The number of samples per step (=stride). See Warnings.
    n_samples_win : int
        The number of samples per window. See Warnings.
    load_y_func : callable, optional
        Function to load the target data. Defaults to None.
    load_y_args : dict, optional
        Parameters to pass to `load_y_func`. Defaults to None.
    x_preprocess : callable, optional
        A transformation to apply to the whole x data as a preprocessing step. Defaults to None.
    y_preprocess : callable, optional
        A transformation to apply to the whole y data as a preprocessing step. Defaults to None.
    x_transform : callable, optional
        A transformation to apply to the windowed x data. Defaults to None.
    y_transform : callable, optional
        A transformation to apply to the windowed y data. Defaults to None.
    precompute : bool, optional
        If True, precompute the whole dataset. Very convenient but needs large memory. Defaults to False.

    Warnings
    --------
    `n_samples_win` and `n_samples_step` are used to segment the data after preprocessing. If you downsample
    the data in `x_preprocess` or `y_preprocess`, make sure to adjust these values accordingly.

    Examples
    --------
    See lightouse/scripts/examples/data_loader.ipynb#torch_dataset_with_preprocessing for an example of
    how to use this class.
    """

    def __init__(
        self,
        load_x_func: callable,
        load_x_args: dict,
        n_samples_step: int,
        n_samples_win: int,
        load_y_func: callable = None,
        load_y_args: dict = None,
        x_preprocess: callable = None,
        y_preprocess: callable = None,
        x_transform: callable = None,
        y_transform: callable = None,
        precompute: bool = False,
    ):
        self.x_transform = x_transform
        self.use_y = load_y_func is not None
        self.y_transform = y_transform

        if load_x_args is not None:
            self.x_data = load_x_func(**load_x_args)
            
        else:
            self.x_data = load_x_func()
        if self.use_y:
            self.y_data = load_y_func(**load_y_args)

        self.n_samples_step = n_samples_step
        self.n_samples_win = n_samples_win

        if x_preprocess is not None:
            self.x_data = x_preprocess(self.x_data)
            print(self.x_data.shape)
        if y_preprocess is not None and self.use_y:
            self.y_data = y_preprocess(self.y_data)

        # check if time dimension is the same
        if self.use_y:
            assert (
                self.x_data.shape[-1] == self.y_data.shape[-1]
            ), f"Time dimension (last) mismatch: x={self.x_data.shape[-1]} != {self.y_data.shape[-1]}=y"

        # Time dimension must be the last
        self.ds_len = int((self.x_data.shape[-1] - self.n_samples_win) / self.n_samples_step + 1)

        self.precompute = False
        if precompute:
            logger.info("Precomputing dataset (could take a while)")
            # TODO: parallelize this
            self.precomputed_items = [self.__getitem__(i) for i in tqdm(range(self.ds_len))]
            self.precompute = True

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):
        if self.precompute:
            return self.precomputed_items[idx]
        else:
            start = self.n_samples_step * idx
            stop = (self.n_samples_step * idx) + self.n_samples_win
            x_data = self.x_data[..., start:stop]
            print(f"Before transform: {x_data.shape}")
            if self.x_transform:
                x_data = self.x_transform(x_data)
                print(f"After transform: {x_data.shape}")
            if self.use_y:
                y_data = self.y_data[..., start:stop]
                if self.y_transform:
                    y_data = self.y_transform(y_data)
                return x_data, y_data
            return x_data
