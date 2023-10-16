import os
import pathlib
import torch
import einops
import random
import pandas as pd
import numpy as np
import functools
import datetime
from tqdm import tqdm
import src.utils
from src.utils import MomentEstimator
from collections.abc import Iterable
from scipy.spatial.transform import Rotation as R
import zipfile
import tempfile

import sys


def cached(f):
    """
    Decorator that creates a cached property.
    """
    key = f.__name__

    @property
    @functools.wraps(f)
    def decorated(self):
        if key not in self._cache:
            self._cache[key] = f(self)
        return self._cache[key]

    return decorated


def get_dataset(
    dataset_name,
    dataset_args,
    root_dir,
    config_path,
    num_classes=None,
    skip_files=[],
    label_map=None,
    replace_classes=None,
    test_mode=False,
    valid_mode=False,
    inference_mode=False
):
    allowed_datasets = ['TimeSeries', 'STFT', 'HUNT4Masked']
    if dataset_name in allowed_datasets:
        cls = getattr(
            sys.modules[__name__],
            f'{dataset_name}Dataset'
        )
        return cls(args=dataset_args,
                   root_dir=root_dir,
                   config_path=config_path,
                   num_classes=num_classes,
                   skip_files=skip_files,
                   label_map=label_map,
                   replace_classes=replace_classes,
                   test_mode=test_mode,
                   valid_mode=valid_mode,
                   inference_mode=inference_mode
                  )
    else:
        raise ValueError((f'No Dataset class with name"{dataset_name}".\n'
                          f'Allowed dataset names: {allowed_datasets}'))


class HARDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir,
        x_columns, y_column,
        num_classes,
        padding_val=0.0,
        label_map=None,
        replace_classes=None,
        config_path='',
        skip_files=[],
        test_mode=False,
        valid_mode=False,
        inference_mode=False,
        sep=',',
        header='infer',
    ):
        '''Super class dataset for classification

        Parameters
        ----------
        root_dir (string): Directory of training data
        x_columns (list of str): columns of sensors
        y_column (str): column of label
        num_classes (int): how many classes in dataset
        padding_val (float), optional: Value to insert if padding required
            Note: padding is only applied in inference mode to avoid label
            padding.
        label_map (dict): mapping labels in dataframe to other values
        replace_classes (dict): mapping which replace to replace
        config_path (str): to save normalization params on disk
        skip_files (list of string):
            csv files to in root_dir not to use. If None, all are used
        test_mode (bool): whether dataset used for testing
        valid_mode (bool): whether dataset used for validation
        inference_mode (bool): whether dataset used for inference
            i.e., y is not returned
        sep (str): Which sep used in dataset files
        header (int, list of int, None): Row numbers to use as column names

        '''
        self._cache = {}
        self.root_path = root_dir
        self.x_columns = x_columns
        self.y_column = y_column
        self.num_classes = num_classes
        self.padding_val = padding_val
        self.label_map = label_map
        self.replace_dict = replace_classes
        self.config_path = config_path
        self.skip_files = skip_files
        self.train_mode = not (test_mode or inference_mode or valid_mode)
        self.inference_mode = inference_mode
        self.sep = sep
        self.header = header

    def replace_classes(self, y):
        if self.replace_dict:
            return y.replace(self.replace_dict)
        else:
            return y 

    #@property
    def y(self):
        '''Returns y/true_label tensor(s)

        Returns
        -------
        either tensor or dict of tensors with filename as keys

        '''
        msg = ('Implement y(): Returns y/true_label tensor(s)')
        raise NotImplementedError(msg)

    def post_proc_y(self, t, overlap_kind='mean'):
        '''Undo all changes made in this Dataset class to original y data

        It assumes tensor with shape
        [num_batches, sequence_length, d]
        Depending on subclass, different operations have to
        be performed to achieve correct alignement
        d can be any dimension
        Example: model probability prediction with shape
                [num_batches,sequence_length,num_classes]

        Parameters
        ----------
        t (tensor):
            Has shape [num_batches,sequence_length,d]
        overlap_kind (str), optional:
            What to do with possible overlapping areas. (default is 'mean')
            'sum' adds the values in the overlapping areas
            'mean' computes the mean of the overlapping areas

        Returns
        -------
        either tensor or dict of tensors with filename as keys
            Each tensor's shape: [signal_len, n]

        '''
        msg = ('Implement post_proc_y(): Returns tensor(s) '
               'aligned with original signal')
        raise NotImplementedError(msg)


    @cached
    def _label_cols_available(self):
        '''Is there a y_column in every given root file'''
        filenames = [x for x in os.listdir(self.root_path) \
                     if x not in self.skip_files]
        for fn in tqdm(filenames):
            available_cols = pd.read_csv(
                os.path.join(self.root_path, fn),
                index_col=0,
                nrows=0,
                header=self.header,
                sep=self.sep,
            ).columns.tolist()
            if self.y_column not in available_cols:
                print(f'No label column {self.y_column} in {fn}...'
                      'Skipping labels')
                return False
        return True


class TimeSeriesDataset(HARDataset):
    """Dataset for time series based HAR."""
    def __init__(self, args,
                 root_dir,
                 num_classes,
                 config_path='',
                 label_map=None,
                 replace_classes=None,
                 skip_files=[],
                 test_mode=False,
                 valid_mode=False,
                 inference_mode=False,
                 **kwargs
        ):
        '''Using "raw" time series signals as dataset

        Parameters
        ----------
        root_dir (string): Directory of training data
        x_columns (list of str): columns of sensors
        y_column (str): column of label
        num_classes (int): how many classes in dataset
        padding_val (float), optional: Value to insert if padding required
            Note: padding is only applied in inference mode to avoid label
            padding.
        label_map (dict): mapping labels in dataframe to other values
        replace_classes (dict): which labels to replace
        config_path (str): to save normalization params on disk
        skip_files (list of string):
            csv files to in root_dir not to use. If None, all are used
        test_mode (bool): whether dataset used for testing
        valid_mode (bool): whether dataset used for validation
        inference_mode (bool): whether dataset used for inference
            i.e., y is not returned
        args (dict): Dataset specific parameters
            Needs to include:
            sequence_length (int): Window size in time samples
            frame_shift (int): Window shifting in time samples
            normalize (bool): whether to normalize

        '''
        self.size = 0
        self.sequence_length = args['sequence_length']
        self.frame_shift = args['frame_shift']
        if self.frame_shift is None:
            self.frame_shift = self.sequence_length
        elif self.frame_shift == 'half':
            self.frame_shift = self.sequence_length//2
        self.normalize = args['normalize']
        assert(self.normalize and config_path!='') or not self.normalize, \
                'Config path needs to be provided'
        super().__init__(
            root_dir=root_dir,
            config_path=config_path,
            x_columns=args['x_columns'],
            y_column=args['y_column'],
            padding_val=args['padding_val'],
            num_classes=num_classes,
            label_map=label_map,
            replace_classes=replace_classes,
            skip_files=skip_files,
            test_mode=test_mode,
            valid_mode=valid_mode,
            inference_mode=inference_mode
        )
        self.data = self.read_all(root_dir)
        self.data_ranges = self._get_data_ranges()
        if self.normalize:
            if 'norm_params_path' in args:
                self.normalize_params_path = args['norm_params_path']
            else:
                self.normalize_params_path = os.path.join(
                    config_path,
                    f'normalization_params_TS_feats{self.feature_dim}_seqlen{self.seq_length}'
                )
            force = args['force_norm_comp'] if 'force_norm_comp' in args else False
            force = force and self.train_mode  # Force impossible for test/valid
            if self.normalize != 'minmax':
                self.mean = self._mean(save_on_disk=self.train_mode, force=force)
                self.std = self._std(save_on_disk=self.train_mode, force=force)
            elif self.normalize == 'minmax':
                self._min, self._max = self._min_max(save_on_disk=self.train_mode, force=force)
            self.normalize_data()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        fn = self.get_filename_for_idx(idx)
        # Identify idx in dataframe
        range_start_idx = min(self.data_ranges[fn])
        # start_idx = idx if range_start_idx==0 else idx%range_start_idx
        start_idx = idx-range_start_idx
        start_idx = start_idx * self.frame_shift
        end_idx = start_idx + self.sequence_length
        # Determine window to return:
        x = self.data[fn][0][start_idx:end_idx]
        if self.inference_mode:
            # In inference_mode padding is applied, otherwise shape mismatch
            overflow = abs(min(0, len(self.data[fn][0])-end_idx))
            return torch.nn.functional.pad(
                input=x,
                pad=[0,0,0,overflow],
                value=self.padding_val
            )
        else:
            y = self.data[fn][1][start_idx:end_idx]
            return x, y

    @property
    def seq_length(self):
        '''Input sequence length'''
        return self.sequence_length

    @property
    def feature_dim(self):
        '''Input feature dimensionality'''
        return len(self.x_columns)

    @property
    def output_shapes(self):
        '''Shape of y output if given and one-hot encoded'''
        return self.num_classes

    @property
    def input_shape(self):
        '''Num sensor axes'''
        return self.feature_dim

    #@property
    def y(self):
        '''Returns y_column values as indices

        Returns
        -------
        dict of tensors

        '''
        res = {}
        for k, (_, _y) in self.data.items():
            # res[k] = self.reverse_label_map(_y.numpy())
            res[k] = _y.numpy()
        return res

    def post_proc_y(self, t, overlap_kind='mean'):
        '''Undo all changes made in this Dataset to original y data

        Here, sliding windows are aligned, argmax applied to probabilities
        to get class indices
        values.

        '''
        new_t = self.align(t, overlap_kind)
        new_t = src.utils.argmax(new_t, axis=-1)  # Get classes for preds
        # new_t = {k: self.reverse_label_map(v) for k, v in new_t.items()}
        return new_t

    def align(self, t, overlap_kind='mean'):
        '''Reshapes tensors(e.g. model predictions) to original signal dimension

        Parameters
        ----------
        t (tensor):
            (e.g. model predictions as probabilities for each class)
            Shape: [num_batches,sequence_length,d]
        overlap_kind (str): when overlapping windows have to be considered

        Returns
        -------
        dict of tensors/arrays
            model predictions of dimension [signal_len, d]

        '''
        t_dict = {}
        for filename, _range in self.data_ranges.items():
            _t = t[_range].numpy()
            _t = unfold_windows(
                arr=_t,
                window_size=self.sequence_length,
                window_shift=self.frame_shift
            )
            if self.inference_mode:
                # Cut padded parts at the end
                t_dict[filename] = _t[:self.y()[filename].shape[0]]
            else:
                t_dict[filename] = _t
        return t_dict

    def read_all(self, root_path):
        """ Reads all csv files in a given path"""
        data = {}
        filenames = [x for x in os.listdir(root_path) \
                     if x not in self.skip_files]
        uc = self.x_columns if not self._label_cols_available \
                            else self.x_columns+[self.y_column]
        for fn in tqdm(filenames):
            df = pd.read_csv(
                os.path.join(root_path, fn),
                usecols=uc
            )
            # Required for classification
            if self._label_cols_available and self.label_map is not None:
                df[self.y_column] = df[self.y_column].apply(
                    lambda _x: self.label_map[_x]
                )
            x = torch.tensor(df[self.x_columns].values,dtype=torch.float32)
            if self._label_cols_available:
                df[self.y_column] = self.replace_classes(df[self.y_column])
                y = torch.tensor(df[self.y_column].values,
                                 dtype=torch.int64)
                data[fn] = (x, y)
            else:
                data[fn] = (x, None)
        return data

    def _get_data_ranges(self):
        '''To identify which subj to use given idx'''
        data_ranges = {}
        for fn, (x, y) in self.data.items():
            num_slices = get_num_slices(
                total_amount=len(x),
                sequence_length=self.sequence_length,
                frame_shift=self.frame_shift,
                padding=self.inference_mode
            )
            self.size += num_slices
            data_ranges[fn] = range(self.size-num_slices, self.size)
        return data_ranges

    def get_filename_for_idx(self, idx):
        '''Given idx, which filename to use'''
        return [fn for fn, r in self.data_ranges.items() if idx in r][0]

    def normalize_data(self):
        '''Normalize time signals'''
        for fn, (x,y) in self.data.items():
            if self.normalize != 'minmax':
                x = normalize(x=x, mean=self.mean, std=self.std)
            elif self.normalize == 'minmax':
                x = normalize_min_max(x, self._min, self._max)
            self.data[fn] = (x,y)

    # @cached
    def _mean(self, save_on_disk=False, force=False):
        '''Mean across all samples for each feature

        If mean not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed mean in normalize_params_path
        force (bool): Force recomuptation even if already on disk

        Returns
        -------
        torch.Tensor

        '''
        _m_path = os.path.join(self.normalize_params_path, 'mean.csv')
        if not os.path.exists(_m_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_m_path}'
                )
            else:
                _sum = sum([x.sum(axis=0) for x,_ in self.data.values()])
                _len = sum([x.shape[0] for x,_ in self.data.values()])
                _m = _sum/_len
                if save_on_disk:
                    src.utils.store_tensor(_m, _m_path)
        else:
            _m = src.utils.load_tensor(_m_path)
        return _m

    # @cached
    def _std(self, save_on_disk=False, force=False):
        '''Std across all samples for each feature

        If std not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed mean in normalize_params_path
        force (bool): Force recomuptation even if already on disk

        Returns
        -------
        torch.Tensor

        '''
        _s_path = os.path.join(self.normalize_params_path, 'std.csv')
        if not os.path.exists(_s_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_s_path}'
                )
            else:
                _m = self.mean
                _sum = sum([((x-_m)**2).sum(axis=0) for x,_ in self.data.values()])
                _len = sum([x.shape[0] for x,_ in self.data.values()])
                _s = np.sqrt(_sum/_len)
                if save_on_disk:
                    src.utils.store_tensor(_s, _s_path)
        else:
            _s = src.utils.load_tensor(_s_path)
        _s = torch.where(_s==0.0, EPS, _s)  # Avoid div by 0
        return _s

    def _min_max(self, save_on_disk=False, force=False):
        '''Min and max across all samples for each feature

        If min or max not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed mean in normalize_params_path
        force (bool): Force recomuptation even if already on disk

        Returns
        -------
        torch.Tensor

        '''
        _min_path = os.path.join(self.normalize_params_path, 'min.csv')
        _max_path = os.path.join(self.normalize_params_path, 'max.csv')
        if not os.path.exists(_min_path) or not os.path.exists(_max_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_min_path} or {_max_path}'
                )
            else:
                _min = torch.stack([x.min(axis=0)[0] for x,_ in self.data.values()]).min(axis=0)[0]
                _max = torch.stack([x.max(axis=0)[0] for x,_ in self.data.values()]).max(axis=0)[0]
                if save_on_disk:
                    src.utils.store_tensor(_min, _min_path)
                    src.utils.store_tensor(_max, _max_path)
        else:
            _min = src.utils.load_tensor(_min_path)
            _max = src.utils.load_tensor(_max_path)
        return _min, _max


class STFTDataset(HARDataset):
    """Dataset for spectrogram-based HAR."""

    def __init__(self, args,
                 root_dir,
                 num_classes,
                 config_path='',
                 label_map=None,
                 replace_classes=None,
                 skip_files=[],
                 test_mode=False,
                 valid_mode=False,
                 inference_mode=False,
                 **kwargs
        ):
        '''Using spectrograms of time series signals as dataset

        Parameters
        ----------
        root_dir (string): Directory of training data
        x_columns (list of str): columns of sensors
        y_column (str): column of label
        num_classes (int): how many classes in dataset
        padding_val (float), optional: Value to insert if padding required
            Note: padding is only applied in inference mode to avoid label
            padding.
        label_map (dict): mapping labels in dataframe to other values
        replace_classes (dict): mapping which classes to replace
        config_path (str): to save normalization params on disk
        skip_files (list of string):
            csv files to in root_dir not to use. If None, all are used
        test_mode (bool): whether dataset used for testing
        valid_mode (bool): whether dataset used for validation
        inference_mode (bool): whether dataset used for inference
            i.e., y is not returned
        args (dict): Dataset specific parameters
            Needs to include:
                n_fft (int): STFT window size
                hop_length (int): STFT window shift
                normalize (bool): Whether to normalize spectrograms
                phase (bool): Include phase

        '''
        self.normalize = args['normalize']
        self.stack_axes = True
        self.size = 0
        self._y = {}
        # Read file params
        self.drop_labels = args['drop_labels'] if 'drop_labels' in args else []
        # In case of resampling
        self.source_freq = args['source_freq'] if 'source_freq' in args else 50
        self.target_freq = args['target_freq'] if 'target_freq' in args else 50
        # Freq domain split
        self.n_fft = args['n_fft']
        self.hop_length = args['hop_length']
        self.hop_length = self.n_fft//2 if args['hop_length'] is None else args['hop_length']
        self.window = torch.hann_window(self.n_fft)
        self.phase = args['phase']
        # Time domain split
        assert args['sequence_length'] > self.hop_length, \
                print('sequence_length < hop_length not allowed')
        self.sequence_length = args['sequence_length'] // self.hop_length -1
        frame_shift = args['frame_shift']
        if frame_shift is None:
            frame_shift = args['sequence_length']
        elif frame_shift == 'half':
            frame_shift = args['sequence_length']//2
        self.frame_shift = frame_shift // self.hop_length
        # Windowed labels handling
        self.windowed_labels_kind = args['windowed_labels_kind'] \
                if 'windowed_labels_kind' in args else 'argmax'
        super().__init__(
            root_dir=root_dir,
            config_path=config_path,
            x_columns=args['x_columns'],
            y_column=args['y_column'],
            padding_val=args['padding_val'],
            num_classes=num_classes,
            label_map=label_map,
            replace_classes=replace_classes,
            skip_files=skip_files,
            test_mode=test_mode,
            valid_mode=valid_mode,
            inference_mode=inference_mode,
            sep=args['sep'] if 'sep' in args else ',',
            header=args['header'] if 'header' in args else 'infer',
        )
        self.data = self.read_all(root_dir)
        self.data_ranges = self._get_data_ranges()
        if self.normalize:
            if 'norm_params_path' in args:
                self.normalize_params_path = args['norm_params_path']
            else:
                self.normalize_params_path = os.path.join(
                    config_path,
                    f'normalization_params_STFT_feats{self.feature_dim}_seqlen{self.seq_length}'
                )
            force = args['force_norm_comp'] if 'force_norm_comp' in args else False
            force = force and self.train_mode  # Force impossible for test/valid
            self.mean = self._mean(save_on_disk=self.train_mode, force=force)
            self.std = self._std(save_on_disk=self.train_mode, force=force)
            self.normalize_data()

    def __getitem__(self, idx):
        fn = self.get_filename_for_idx(idx)
        # Identify idx in dataframe
        range_start_idx = min(self.data_ranges[fn])
        start_idx = idx-range_start_idx
        start_idx = start_idx * self.frame_shift
        end_idx = start_idx + self.sequence_length
        win_len = end_idx - start_idx
        # Determine window to return:
        x = self.data[fn][0][start_idx:end_idx]
        if self.inference_mode:
            if len(x) != len(self.data[fn][0]):
                # In inference_mode padding is applied, otherwise shape mismatch
                overflow = abs(min(0, len(self.data[fn][0])-end_idx))
                x = torch.nn.functional.pad(
                    input=x,
                    pad=[0,0,0,overflow],
                    value=self.padding_val
                )
            return x
        else:
            y = self.data[fn][1][start_idx:end_idx]
            return x, y

    def __len__(self):
        return self.size

    @property
    def seq_length(self):
        '''Input sequence length'''
        return self.sequence_length

    @property
    def feature_dim(self):
        '''Input feature dimensionality'''
        if not self.phase:
            return (self.n_fft // 2 + 1) * len(self.x_columns)
        else:
            return (self.n_fft // 2 + 1) * len(self.x_columns) * 2

    @property
    def output_shapes(self):
        '''Shape of y output if given and one-hot encoded'''
        return self.num_classes

    @property
    def input_shape(self):
        '''Num bins'''
        return self.feature_dim

    #@property
    def y(self, return_probs=False, probs_aggr_window_len=None):
        '''Returns y_column values as indices or probabilities

        Parameters
        ----------
        return_probs (bool, optional): Compute probabilities
        probs_aggr_window_len (int, optional): Window length for probs

        Returns
        -------
        dict of tensors

        '''
        if return_probs:
            if probs_aggr_window_len:
                aggr_len = probs_aggr_window_len
                aggr_shift = probs_aggr_window_len
            else:
                aggr_len = self.n_fft
                aggr_shift = self.hop_length
            new_y = {}
            for fn, y_true in self._y.items():
                new_y[fn] = windowed_labels(
                    labels=y_true,
                    num_labels=self.num_classes,
                    frame_length=aggr_len,
                    frame_step=aggr_shift,
                    pad_end=True,
                    kind='density'
                )
            return new_y
        return self._y

    def post_proc_y(self, t, overlap_kind='mean', return_probs=False, probs_aggr_window_len=None):
        '''Undo all changes made in this Dataset to original y data

        Here, sliding windows are aligned 2 or 3 times:
        1 for normal splitting, 1 for STFT computation, and 1 if resampling done.
        argmax applied to probabilities to get class indices values.

        Parameters
        ----------
        t (array like): tensor to process
        overlap_kind (str, optional): How to handle overlaps when unfolding
        return_probs (bool, optional): Do not apply argmax if True
        probs_aggr_window_len (int, optional): aggregate probs if not None

        '''
        t_dict = {}
        for filename, _range in self.data_ranges.items():
            try:
                _t = t[_range].numpy()
            except TypeError:
                _t = np.array(t[slice(_range.start,_range.stop)][0])
            # Split to spectrograms dim
            if _t.shape[0] != 1:
                _t = unfold_windows(
                    arr=_t,
                    window_size=self.sequence_length,
                    window_shift=self.frame_shift,
                    overlap_kind=overlap_kind
                )
            else:
                _t = _t[0]
            if self.inference_mode:
                # Cut padded parts at the right side
                overflow = abs(min(0, len(self.data[filename][0])-len(_t)))
                _t = _t[:len(_t)-overflow]
            if return_probs:
                if probs_aggr_window_len:
                    amount_to_inlude = get_num_slices(
                        total_amount=probs_aggr_window_len,
                        sequence_length=self.n_fft,
                        frame_shift=self.hop_length,
                        padding=False
                    )
                    amount_to_shift = get_num_slices(
                        total_amount=probs_aggr_window_len,
                        sequence_length=self.n_fft,
                        frame_shift=self.hop_length,
                        padding=True
                    )
                    new_t = []
                    for i in range(0, len(_t), amount_to_shift):
                        cutted_probs = _t[i:i+amount_to_inlude]
                        new_t.append(cutted_probs.mean(axis=0))
                    _t = np.array(new_t)
                t_dict[filename] = _t
                continue
            # Spectrograms dim to time dim
            _t = unfold_windows(
                arr=_t,
                window_size=self.n_fft,
                window_shift=self.hop_length,
                overlap_kind=overlap_kind
            )
            # Undo resampling if required
            if self.source_freq != self.target_freq:
                df_t = pd.DataFrame(_t)
                _t = src.utils.resample(
                    signal=df_t,
                    source_rate=self.target_freq,
                    target_rate=self.source_freq,
                    discrete_columns=df_t.columns,
                    resampler='fourier',
                    padder=None,
                    pad_size=None
                ).values
            if self.inference_mode:
                # Cut padded parts at the end
                t_dict[filename] = _t[:self.y()[filename].shape[0]]
            else:
                t_dict[filename] = _t
        if not return_probs:
            t_dict = src.utils.argmax(t_dict, axis=-1)  # Get classes for preds
        return t_dict

    def read_all(self, root_path):
        """ Reads all csv files in a given path and computes STFT"""
        data = {}
        filenames = [x for x in os.listdir(root_path) \
                     if x not in self.skip_files]
        uc = self.x_columns+[self.y_column]
        for fn in tqdm(filenames):
            df = pd.read_csv(
                os.path.join(root_path, fn),
                sep=self.sep,
                usecols=uc,
                header=self.header,
            )
            for drop_label in self.drop_labels:
                df = df[df[self.y_column]!=drop_label]
            df = df.dropna()  # Drop nan values
            # Required for classification
            if self._label_cols_available:
                df[self.y_column] = self.replace_classes(df[self.y_column])
            if self._label_cols_available and self.label_map is not None:
                df[self.y_column] = df[self.y_column].apply(
                    lambda _x: self.label_map[_x]
                )
                self._y[fn] = df[self.y_column].values
            # Resampling if required
            if self.source_freq != self.target_freq:
                discrete_columns=[self.y_column]
                df = src.utils.resample(
                    signal=df,
                    source_rate=self.source_freq,
                    target_rate=self.target_freq,
                    discrete_columns=discrete_columns,
                    resampler='fourier',
                    padder=None,
                    pad_size=None
                )
            x = torch.tensor(df[self.x_columns].values,dtype=torch.float32)
            # reshape required for correct STFT computation:
            # [signal_len, num_channels] -> [num_channels, signal_len]
            x = einops.rearrange(x, 'S C -> C S')
            if self.inference_mode:
                # Padding to make STFT computation easier
                overflow = np.floor((x.shape[-1]-1)/self.hop_length)
                overflow = int(overflow*self.hop_length + self.n_fft)
                overflow = abs(min(0, x.shape[-1]-overflow))
                x = torch.nn.functional.pad(
                    input=x,
                    pad=[0,overflow],
                    value=self.padding_val
                )
            x = torch.stft(
                input=x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=False,
                return_complex=True
            )  # [num_channels, num_bins, num_frames]
            x_cartesian = src.utils.complex_to_cartesian(x)
            x_magnitude = src.utils.complex_to_magnitude(x, expand=True)
            x = x_cartesian if self.phase else x_magnitude
            if self.stack_axes:
                # Stack all spectrograms and put time dim first:
                # [num_channels, num_bins, num_frames, stft_parts] ->
                # [num_frames, num_channels x num_bins x stft_parts]
                x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
            else:
                x = einops.rearrange(x, 'C F T P -> T C F P')
            if self._label_cols_available:
                y = windowed_labels(
                    labels=df[self.y_column].values,
                    num_labels=self.num_classes,
                    frame_length=self.n_fft,
                    frame_step=self.hop_length,
                    pad_end=self.inference_mode,
                    kind=self.windowed_labels_kind
                )
                y_dtype = torch.int64 \
                        if self.windowed_labels_kind=='argmax' \
                        else torch.float32
                y = torch.tensor(y, dtype=y_dtype)
                data[fn] = (x, y)
            else:
                data[fn] = (x, None)
        return data

    def _get_data_ranges(self):
        '''To identify which subj to use given idx'''
        data_ranges = {}
        for fn, (x, y) in self.data.items():
            num_slices = get_num_slices(
                total_amount=len(x),
                sequence_length=self.sequence_length,
                frame_shift=self.frame_shift,
                padding=self.inference_mode
            )
            self.size += num_slices
            data_ranges[fn] = range(self.size-num_slices, self.size)
        return data_ranges

    def get_filename_for_idx(self, idx):
        '''Given idx, which filename to use'''
        return [fn for fn, r in self.data_ranges.items() if idx in r][0]


    def normalize_data(self):
        '''Normalize time signals'''
        for fn, (x,y) in self.data.items():
            x = normalize(x=x, mean=self.mean, std=self.std)
            self.data[fn] = (x,y)

    # @cached
    def _mean(self, save_on_disk=False, force=False):
        '''Mean across all samples for each feature

        If mean not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed mean in normalize_params_path
        force (bool): Force recomputation of mean even if saved on disk

        Returns
        -------
        torch.Tensor

        '''
        _m_path = os.path.join(self.normalize_params_path, 'mean.csv')
        if not os.path.exists(_m_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_m_path}'
                )
            else:
                print('Creating mean...')
                _sum = sum([x.sum(axis=0) for x,_ in self.data.values()])
                _len = sum([x.shape[0] for x,_ in self.data.values()])
                _m = _sum/_len
                if save_on_disk:
                    src.utils.store_tensor(_m, _m_path)
        else:
            _m = src.utils.load_tensor(_m_path)
        return _m

    # @cached
    def _std(self, save_on_disk=False, force=False):
        '''Std across all samples for each feature

        If std not already saved on disk in self.normalize_params_path,
        it is computed using self.data. Otherwise, it is read from
        disk.

        Parameters
        ----------
        save_on_disk (bool): Stores computed std in normalize_params_path
        force (bool): Force recomputation of std even if saved on disk

        Returns
        -------
        torch.Tensor

        '''
        _s_path = os.path.join(self.normalize_params_path, 'std.csv')
        if not os.path.exists(_s_path) or force:
            if not self.train_mode:
                raise FileNotFoundError(
                    f'No normalization param found {_s_path}'
                )
            else:
                print('Creating std...')
                _m = self.mean
                _sum = sum([((x-_m)**2).sum(axis=0) for x,_ in self.data.values()])
                _len = sum([x.shape[0] for x,_ in self.data.values()])
                _s = np.sqrt(_sum/_len)
                if save_on_disk:
                    src.utils.store_tensor(_s, _s_path)
        else:
            _s = src.utils.load_tensor(_s_path)
        _s = torch.where(_s==0.0, EPS, _s)  # Avoid div by 0
        return _s

    def collate_fn(self, data):
        '''Custom collate_fn for different sequence lengths in a batch'''
        x = torch.nn.utils.rnn.pad_sequence([d[0] for d in data],
                                            batch_first=True,
                                            padding_value=0.0)
        y = torch.nn.utils.rnn.pad_sequence([d[1] for d in data],
                                            batch_first=True,
                                            padding_value=0)
        # 0s where padding applied
        mask = torch.ones(y.shape[:2])
        for i in range(len(mask)):
            mask[i][len(data[i][1]):] = 0.0
        return [x, y, mask]


class HUNT4Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        columns,
        sequence_length,
        STFT=False,
        stft_n_fft=None,
        stft_hop_len=None,
        normalize=False,
        config_path='',
        skip_night=False,
        skip_files=None,
        num_samples_per_subj=None,
        num_hours=None,
        sample_freq=50,
        phase=False,
        stack_axes=True,  # Stack STFT sensor axes
        already_STFT=False,
        hour_wise_norm=True,
        norm_params_path=None,
        random_rotations=False,
        return_rotation_angles=False
    ):
        '''Returns HUNT4 arrays stored in root_dir'''
        self._cache = {}
        self.root_path = root_dir
        self.columns = columns
        self.sequence_length = sequence_length
        self.sample_freq = sample_freq
        self.STFT = STFT
        self.n_fft = stft_n_fft
        self.hop_length = self.n_fft//2 if stft_hop_len is None else stft_hop_len
        self.hop_length_in_sec = self.hop_length / sample_freq
        self.normalize = normalize
        self.skip_night = skip_night
        self.skip_files = skip_files
        self.num_samples_per_subj = num_samples_per_subj
        self.num_hours = num_hours
        self.phase = phase
        self.stack_axes = stack_axes
        self.paths = self.get_paths() if not num_hours==0 else []
        self.already_STFT = already_STFT
        self.random_rotations = random_rotations
        self.rotated_angles = {}
        if normalize and self.__len__()!=0:
            if norm_params_path is None:
                norm_params_path = os.path.join(
                    config_path,
                    f'normalization_params_HUNT4_feats{self.feature_dim}_seqlen{self.seq_length}'
                )
            self.mean, self.std = self.get_mean_std(
                save_on_disk=True,
                params_path=norm_params_path,
                hour_wise_norm=hour_wise_norm
            )

    def __getitem__(self, idx):
        x = self._get_x(idx)
        if self.normalize:
            x = normalize(x, self.mean, self.std)
        return x

    def _get_x(self, idx, paths=None):
        '''Get x without normalization'''
        paths = paths if paths else self.paths
        x = self.read(paths[idx])
        if self.random_rotations:
            x, angles = self._rotate_x_randomly(x)
            self.rotated_angles[idx] = angles
        x = self.stft(x) if self.STFT else x
        return x

    def _rotate_x_randomly(self, x):
        '''Rotates 3D accelerometer signals by random roll, pitch, yaw'''
        # Random rotation angles in radians (TODO: think about different distrib)
        roll, pitch, yaw = np.random.uniform(0.0, 2*np.pi, 3)
        r = R.from_euler('xyz', [roll,pitch,yaw], degrees=False)
        rx = []
        for i in range(len(self.columns)//3):
            rx.append(torch.tensor(r.apply(x[:,i*3:(i+1)*3]), dtype=torch.float32))
        x = torch.concat(rx, axis=1)
        return x, [roll, pitch, yaw]

    def __len__(self):
        return len(self.paths)

    @property
    def seq_length(self):
        '''Input sequence length'''
        return self.sequence_length//self.hop_length-1 \
                if self.STFT else self.sequence_length

    @property
    def feature_dim(self):
        '''Input feature dimensionality'''
        if self.STFT:
            if not self.phase:
                return (self.n_fft//2+1)*len(self.columns)
            else:
                return (self.n_fft//2+1)*len(self.columns)*2
        else:
            return len(self.columns)

    @property
    def input_shape(self):
        return self.feature_dim

    @cached
    def used_files(self):
        return self.paths

    #@cached
    def get_paths(self):
        """Gets all files in root_path as list

        Files are assumed to be on the form `ds_path/subject/file.npz`

        """
        print('Getting paths...')
        def _apply_callbacks(x):
            for c in callbacks:
                x = c(x)
            return x
        def _check(x):
            res = []
            for i in x:
                # Correct file format
                if not i.name.endswith('.npz'): continue
                # Night time filter
                if self.skip_night and self._is_night_time(str(i)):
                    continue
                res.append(i)
            return res
        callbacks = [_check]
        res_paths = [
            str(path)
            for subject in tqdm(pathlib.Path(self.root_path).iterdir())
            for path in _apply_callbacks(subject.iterdir())
        ]
        if self.skip_files:
            #res_paths = [p for p in res_paths if p not in self.skip_files]
            res_paths = list(pd.DataFrame(res_paths+self.skip_files).drop_duplicates(keep=False)[0].values)
        if self.num_samples_per_subj:
            res_paths = self._sample_num_samples_per_subj(res_paths)
        if self.num_hours is not None:
            res_paths = self._sample_hours(res_paths)
        return res_paths

    def _all_paths(self):
        """Gets all files in root_path as list

        Same as get_paths but no filtering applied
        Files are assumed to be on the form `ds_path/subject/file.npz`

        """
        print('Getting all paths...')
        def _apply_callbacks(x):
            for c in callbacks:
                x = c(x)
            return x
        def _check(x):
            res = []
            for i in x:
                # Correct file format
                if not i.name.endswith('.npz'): continue
                res.append(i)
            return res
        callbacks = [_check]
        res_paths = [
            str(path)
            for subject in tqdm(pathlib.Path(self.root_path).iterdir())
            for path in _apply_callbacks(subject.iterdir())
        ]
        return res_paths

    def _is_night_time(self, path, _from=0, _to=6):
        '''Is time given in filename during night time (12am-6am)?'''
        time = int(path.split('/')[-1][:-4].split('_')[1])
        if _to < _from:
            return time >= _from or time <= _to
        return _from <= time <= _to

    def _sample_num_samples_per_subj(self, paths):
        res = []
        subj_paths = []
        paths.sort()
        for i in range(len(paths)-1):
            e0, e1 = paths[i:i+2]
            if os.path.dirname(e0) == os.path.dirname(e1):
                subj_paths.append(e0)
            else:
                res += self._sample_files(subj_paths)
                subj_paths = []
        return res

    def _sample_files(self, subj_paths):
        '''Samples num_samples_per_subj from subj_paths if available'''
        if len(subj_paths) < self.num_samples_per_subj: return []
        return random.sample(subj_paths, self.num_samples_per_subj)

    def _sample_hours(self, file_list):
        to_sample = int((self.num_hours*60*60*self.sample_freq)//self.sequence_length)
        return random.sample(file_list, to_sample)

    def get_filename_for_idx(self, idx):
        fn = self.paths[idx].split('/')[-2]
        fn = fn[:-3] if fn.endswith('.7z') else fn
        return fn

    def read(self, path):
        x = np.load(path)['arr_0'][:self.sequence_length,self.columns]
        return torch.tensor(x, dtype=torch.float32)

    def stft(self, x):
        if not self.already_STFT:
            # In case data is not stored already in STFT form:
            # reshape required for correct STFT computation:
            # [signal_len, num_channels] -> [num_channels, signal_len]
            x = einops.rearrange(x, 'S C -> C S')
            x = torch.stft(
                input=x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft),
                center=False,
                return_complex=True
            )  # [num_channels, num_bins, num_frames]
            x_cartesian = src.utils.complex_to_cartesian(x)
            x_magnitude = src.utils.complex_to_magnitude(x, expand=True)
            x = x_cartesian if self.phase else x_magnitude
            if self.stack_axes:
                # Stack all spectrograms and put time dim first:
                # [num_channels, num_bins, num_frames, stft_parts] ->
                # [num_frames, num_channels x num_bins x stft_parts]
                x = einops.rearrange(x, 'C F T P -> T (C F P)')  # P=2
            else:
                x = einops.rearrange(x, 'C F T P -> T C F P')
            return x
        else:
            # In case data is already stored in STFT form:
            if self.stack_axes:
                x = einops.rearrange(x, 'T C F P -> T (C F P)')  # P=2
            return x

    def get_mean_std(self, save_on_disk=True, params_path='', hour_wise_norm=True):
        _m_path = os.path.join(params_path, 'mean.csv')
        _s_path = os.path.join(params_path, 'std.csv')
        if not os.path.exists(_m_path) or not os.path.exists(_s_path):
            if not hour_wise_norm:
                paths = self._all_paths()
                #paths = self.paths
                print('Creating mean...')
                _sums = []
                _lens = []
                for ii in tqdm(range(len(paths))):
                    x = self._get_x(ii, paths=paths)
                    _sums.append(x.sum(axis=0))
                    _lens.append(x.shape[0])
                _sums = sum(_sums)
                _lens = sum(_lens)
                _m = _sums/_lens
                print('Creating std...')
                _sums = []
                _lens = []
                for ii in tqdm(range(len(paths))):
                    x = self._get_x(ii, paths=paths)
                    _sums.append(((x-_m)**2).sum(axis=0))
                    _lens.append(x.shape[0])
                _sums = sum(_sums)
                _lens = sum(_lens)
                _s = np.sqrt(_sums/_lens)
            else:
                paths = self._all_paths()
                moment_shape = self._get_x(0, paths=paths).shape[1:]
                MOMENTS = [
                    MomentEstimator(shape=moment_shape, moments=[1, 2])
                    for _ in range(24)
                ]
                print(f'Computing moments of {self.root_path} into {params_path}')
                #for ii in tqdm(range(self.__len__())):
                for ii in tqdm(range(len(paths))):
                    x = self._get_x(ii, paths=paths)
                    hour = os.path.splitext(os.path.basename(paths[ii]))[0]
                    hour = datetime.datetime.strptime(hour, '%Y-%m-%d_%H').hour
                    MOMENTS[hour].update(x)
                MOMENTS_AGG = MomentEstimator.merge(MOMENTS)
                _m = MOMENTS_AGG.get(1)
                _s = (MOMENTS_AGG.get(2) - MOMENTS_AGG.get(1) ** 2) ** 0.5
            if save_on_disk:
                # On disk we save the stacked version
                _m_to_store = _m if self.stack_axes \
                        else einops.rearrange(_m,  'C F P -> (C F P)')
                src.utils.store_tensor(_m_to_store, _m_path)
                _s_to_store = _s if self.stack_axes \
                        else einops.rearrange(_s,  'C F P -> (C F P)')
                src.utils.store_tensor(_s_to_store, _s_path)
        else:
            _m = src.utils.load_tensor(_m_path)
            _m = _m if self.stack_axes \
                    else torch.reshape(_m, self._get_x(0).shape[1:])
            _s = src.utils.load_tensor(_s_path)
            _s = _s if self.stack_axes \
                    else torch.reshape(_s, self._get_x(0).shape[1:])
        _s = torch.where(_s==0.0, EPS, _s)  # Avoid div by 0
        return _m, _s


class HUNT4MaskedDataset(HUNT4Dataset):
    def __init__(self, args,
                 root_dir,
                 config_path,
                 test_mode,
                 valid_mode,
                 skip_files=[],
                 **kwargs):
        '''Dataset for upstream task of predicting masked STFTs

        Parameters
        ----------
        root_dir (str): Path to HUNT4 data
            Needs to contain subject folders and each subject folder has
            to contain 6-dim time signal files (f.e. sensor axis)
        config_path (str)
        skip_files (list of str): what files/subjects to skip
        args (dict): Dataset specific parameters
            Needs to include:
                x_columns (list of int): Which columns to use
                    Indices required: 0=Bx, 1=By, 2=Bz, 3=Tx, 4=Ty, 5=Tz
                sequence_length (int): Window size in time samples
                normalize (bool): Whether to normalize the data
                STFT (bool): Whether to compute STFT
                stft_n_fft, stft_hop_len (int,int):
                    Fourier transform window size and shift
                skip_night (bool): Data from 12am-6am not considered
                num_hours (int): How many HUNT4 hours to return
                time_alter_percentage (float): time frames to mask in STFT
                time_alter_width (int): num consec. time frames to mask
                time_swap (bool): Include time window swapping like in (Liu et al. 2021)
                time_alter_range tuple(int,int): Limited range to mask
                time_mask_flip_prob(float): probability of flipping time mask
                freq_alter_percentage (float): frequency bins to mask
                freq_alter_width (int): num consec. frequ. bins to mask
                freq_mask_flip_prob(float): probability of flipping freq mask
                stored_as_STFT (bool): is dataset stored as spectrograms
                hour_wise_norm (bool): Weight normaliz. by amount of hours

        '''
        assert not (test_mode and valid_mode), \
                print('test_mode and valid_mode cannot both be True')
        self.time_alter_percentage = args['time_alter_percentage']
        self.time_alter_width = args['time_alter_width']
        self.time_swap = args['time_swap'] if 'time_swap' in args else True
        self.time_alter_range = args['time_alter_range'] \
                if 'time_alter_range' in args else None
        self.time_mask_flip_prob = args['time_mask_flip_prob'] \
                if 'time_mask_flip_prob' in args else 0.0
        self.freq_alter_percentage = args['freq_alter_percentage']
        self.freq_alter_width = args['freq_alter_width']
        self.freq_mask_flip_prob = args['freq_mask_flip_prob'] \
                if 'freq_mask_flip_prob' in args else 0.0
        if test_mode:
            n_hours = args['num_test_hours']
        elif valid_mode:
            n_hours = args['num_valid_hours']
        else:
            n_hours = args['num_train_hours']
        if 'all_columns' in args:
            cols = args['all_columns']
        else:
            cols = []
            for c in args['x_columns']+args['y_columns']:
                if c not in cols: cols.append(c)
        self.x_cols = args['x_columns']
        self.y_cols = args['y_columns']
        self.switch_sensor_prob = args['switch_sensor_prob']
        assert self.switch_sensor_prob==0 or \
            (len(self.x_cols)==len(self.y_cols)), \
            'x_columns and y_columns need same len if switch_sensor_prob>0'
        already_STFT = args['stored_as_STFT'] \
                if 'stored_as_STFT' in args else False
        hour_wise_norm = args['hour_wise_norm'] \
                if 'hour_wise_norm' in args else True
        self.max_perc_to_cut = args['max_perc_to_cut'] \
                if 'max_perc_to_cut' in args else 0.0
        norm_params_path = args['norm_params_path'] \
                if 'norm_params_path' in args else None
        self.return_input_sensor_pos = args['return_input_sensor_pos'] \
                if 'return_input_sensor_pos' in args else False
        self.random_rotations = args['random_rotations'] \
                if 'random_rotations' in args else False
        self.return_rotation_angles = args['return_rotation_angles'] \
                if 'return_rotation_angles' in args else False
        super().__init__(
            root_dir=root_dir,
            columns=cols,
            sequence_length=args['sequence_length'],
            STFT=args['STFT'],
            stft_n_fft=args['stft_n_fft'],
            stft_hop_len=args['stft_hop_len'],
            normalize=args['normalize'],
            config_path=config_path,
            skip_night=args['skip_night_time'],
            skip_files=skip_files,
            num_samples_per_subj=None,
            num_hours=n_hours,
            sample_freq=args['sample_freq'],
            stack_axes=False,  # Stacking applied after masking
            phase=args['phase'],
            already_STFT=already_STFT,
            hour_wise_norm=hour_wise_norm,
            norm_params_path=norm_params_path,
            random_rotations=self.random_rotations
        )

    def __getitem__(self, idx):
        ''' Masked signal (x), clean signal (y), and corresp. mask (m)

        Mask and y are returned as combined tuple, which is considered
        in loss computation.

        '''
        x = super(HUNT4MaskedDataset, self).__getitem__(idx)
        y = x.detach().clone()
        # Maybe swap sensors
        if torch.rand(1) > self.switch_sensor_prob:
            x = x[:,self.x_cols,:,:]
            y = y[:,self.y_cols,:,:]
            in_sensor_pos = self._get_input_sensor_position(self.x_cols)
        else:
            x = x[:,self.y_cols,:,:]
            y = y[:,self.x_cols,:,:]
            in_sensor_pos = self._get_input_sensor_position(self.y_cols)
        loss_mask = torch.ones_like(y)
        # Maybe mask out / swap time steps
        if self.time_alter_percentage > 0 and self.time_alter_width > 0:
            if self.time_swap:
                x, mask = liu_et_al_mask_swap(
                    x=x,
                    p=self.time_alter_percentage,
                    width=self.time_alter_width,
                    axis=0,
                    limit_range=self.time_alter_range
                )
            else:
                x, mask = randomly_mask_tensor(
                    x=x,
                    p=self.time_alter_percentage,
                    width=self.time_alter_width,
                    axis=0,
                    prob=0.9,
                    limit_range=self.time_alter_range,
                    flip_prob=self.time_mask_flip_prob
                )
            loss_mask *= einops.rearrange(mask, 'T -> T 1 1 1')
        if self.freq_alter_percentage > 0 and self.freq_alter_width > 0:
            x, mask = randomly_mask_tensor(
                x=x,
                p=self.freq_alter_percentage,
                width=self.freq_alter_width,
                axis=2,
                #prob=1.0,
                prob=0.9,
                flip_prob=self.freq_mask_flip_prob
            )
            loss_mask *= einops.rearrange(mask, 'F -> 1 1 F 1')
        loss_mask = 1.0 - loss_mask  # Set 1.0 where changes made
        # Reshape to (#timesteps, #features) by flattening 'C F P'
        x = einops.rearrange(x, 'T C F P -> T (C F P)')
        y = einops.rearrange(y, 'T C F P -> T (C F P)')
        loss_mask = einops.rearrange(loss_mask, 'T C F P -> T (C F P)')
        y = (y, loss_mask)
        if self.return_input_sensor_pos and self.return_rotation_angles:
            y = (y, in_sensor_pos, self._get_applied_rotation_angles(idx))
        elif self.return_input_sensor_pos:
            y = (y, in_sensor_pos)
        elif self.return_rotation_angles:
            y = (y, self._get_applied_rotation_angles(idx))
        else:
            return x, y

    def _get_input_sensor_position(self, used_columns):
        '''Returns whether the back or thigh sensor are used as input

        Back is encoded as 0
        Thigh is encoded as 1

        '''
        if all([c in [0,1,2] for c in used_columns]):
            # back sensor
            return torch.tensor([0]*self.seq_length, dtype=torch.int64)
        if all([c in [3,4,5] for c in used_columns]):
            # thigh sensor
            return torch.tensor([1]*self.seq_length, dtype=torch.int64)

    def _get_applied_rotation_angles(self, idx):
        '''Get rotation angle applied to sample idx'''
        angles = self.rotated_angles[idx]
        return torch.tensor([angles]*self.seq_length, dtype=torch.float32)

    @property
    def output_shapes(self):
        '''Shape of spectrogram bins is the output shape'''
        fd = super(HUNT4MaskedDataset, self).feature_dim
        fd = fd // (len(self.columns)/len(self.y_cols))
        if self.return_input_sensor_pos and self.return_rotation_angles:
            return [int(fd), 2, 3]
        elif self.return_input_sensor_pos:
            return [int(fd), 2]
        elif self.return_rotation_angles:
            return [int(fd), 3]
        else:
            return int(fd)

    @property
    def input_shape(self):
        '''Depending on x_cols and y_cols feature dim can be different'''
        fd = super(HUNT4MaskedDataset, self).feature_dim
        fd = fd // (len(self.columns)/len(self.x_cols))
        return int(fd)

    def collate_fn(self, data):
        '''Cuts parts of the signal based on max_perc_to_cut

        This simulates that downstream data not always matches upstream
        data sequence lengths

        '''
        # In 50% of cases whole seq_length is used
        if self.max_perc_to_cut == 0 or torch.rand(())<0.5:
            return torch.utils.data.default_collate(data)
        perc_to_keep = 1-np.random.uniform(0,self.max_perc_to_cut)
        width_to_keep = int(self.seq_length*perc_to_keep)
        start_idx = sample_n_choose_k(n=self.seq_length-width_to_keep, k=1)
        end_idx = start_idx + width_to_keep
        new_data = []
        for b in data:
            x = b[0]
            y, loss_mask = b[1]
            new_data.append((x[start_idx:end_idx],
                            (y[start_idx:end_idx], loss_mask[start_idx:end_idx])))
        return torch.utils.data.default_collate(new_data)


###########################################################################
#                Helpful globally accessible functions                    #
###########################################################################

def liu_et_al_mask_swap(x, p, width=1, axis=0, limit_range=None):
    """Randomly swaps and masks a tensor

    See: [Liu et al. 2021]

    Swap random tensor elements along a specified axis

    See `swap_tensor`, except with random `idx_src` and `idx_dst`

    Parameters
    ----------
    x : torch.Tensor<...>
        Input tensor
    p : float
        Fraction of values to swap
    width: int
        Number of contiguous value to swap
    axis: int
        Tensor axis to swap along
    limit_range: tuple(int, int) or None
        Range allowed to be masked. None means:
        limit_range=x.shape[axis]-width (default is None)

    Returns
    -------
    torch.Tensor<...>
    torch.Tensor<length>
    """
    v = torch.rand(())
    total_mask = torch.ones(x.shape[axis])
    if v < 0.8:
        x, mask = randomly_mask_tensor(x=x, p=p, width=width, axis=axis,
                                       limit_range=limit_range)
        total_mask *= mask
    elif v < 0.9:
        x, mask = randomly_swap_tensor(x=x, p=p, width=width, axis=axis,
                                       limit_range=limit_range)
        total_mask *= mask
    return x, total_mask


def randomly_mask_tensor(x, p, width=1, axis=0, prob=1.0, limit_range=None, flip_prob=0.0):
    """Zero out random slices along an axis

    See `mask_tensor` except with a randomized mask
    See also `gen_contiguous_mask`

    Parameters
    ----------
    x : torch.Tensor<...>
        Input tensor
    p : float
        Fraction of values to zero out
    width: int
        Number of contiguous zeros in the mask
    axis: int
        Tensor axis to swap along
    prob: float
        Probability of masking. 1.0 means:
        always masking applied (default=1.0)
    limit_range: tuple(int, int) or None
        Range allowed to be masked. None means:
        limit_range=x.shape[axis]-width (default is None)
    flip_prob: float
        Probability of flipping the mask with torch.flip (default is 0.0)

    Returns
    -------
    torch.Tensor<...>
    torch.Tensor<length>
    """
    v = torch.rand(())
    total_mask = torch.ones(x.shape[axis])
    if v < prob:
        shape = x.shape
        if limit_range is None:
            length = shape[axis]
            n = length - width
        else:
            n = limit_range.copy()
            n[1] = n[1] - width
            length = n[1] - n[0]
        ratio = length / width
        num_starts = int(p * ratio)
        starts = sample_n_choose_k(n=n, k=num_starts)
        mask = gen_contiguous_mask(shape[axis], starts, width)
        if torch.rand(()) < flip_prob:
            mask = torch.flip(mask, dims=(0,))
        x = mask_tensor(x, mask=mask, axis=axis)
        total_mask *= mask
    return x, total_mask


def sample_n_choose_k(n, k):
    """Samples k elements from [0...n) or range without replacement

    Parameters
    ----------
    n : int or tuple(int, int)
        Either max value to sample or start and end index to sample from
    k : int

    Returns
    -------
    torch.Tensor<k>
    """
    if isinstance(n, Iterable):
        return torch.tensor(random.sample(range(*n), k))
    return torch.tensor(random.sample(range(n), k))


def gen_contiguous_mask(length, starts, width=1, dtype=torch.float32):
    """Generates a binary mask with contiguous sets of zeros

    Example:
        length = 10         # Length of mask vector
        starts = [1, 5, 6]  # Start of zero-segments
        width = 2           # Size of zero-segments
    Will result in:
        mask = [1, 0, 0, 1, 1, 0, 0, 0, 1, 1]

    Note that the zero-segments [5, 6] and [6, 7] overlaps

    Parameters
    ----------
    length: int
        Length of resulting mask vector
    starts: torch.Tensor[int]
        Start indices of zero-segments
    width: int
        Size of the zero-segments
    dtype: torch datatype
        Datatype of output mask

    Returns
    -------
    mask : torch.Tensor<length>
    """
    # Generate indices for the positions that will be masked
    indices = torch.reshape(starts[:, None] + torch.arange(0, width)[None],
                            (-1, 1)).to(torch.int64)
    updates = torch.ones(starts.shape[0] * width)
    hits = torch.zeros([length])
    # "Paint" in the updates in an empty (zero) array
    hits = hits.scatter_(0, indices[:,0], updates)
    # The mask should be True/1 wherever nothing was "painted"
    return (hits==0).to(dtype)


def mask_tensor(x, mask, axis=0):
    """Masks out a tensor with a provided mask along a given axis

    Example:
        x = [[0, 1, 2, 3],  # Mask this tensor
             [4, 5, 6, 7]]
        mask = [0, 1]       # Using this mask
        axis = 1            # Along this axis (columns)
    Will result in:
        y = [[0, 0, 0, 0],
             [1, 1, 1, 1]]

    Parameters
    ---------
    x : torch.Tensor<...>
        The tensor to mask (must have rank > 0)
    mask : torch.Tensor<?>
        Multiplicative mask vector (must match x along masking axis)
    axis : int
        The axis to apply the mask along

    Returns
    -------
    torch.Tensor<...>
    """
    shape = x.shape
    length = shape[axis]
    rank = len(shape)
    new_shape = [length if i==axis else 1 for i in range(rank)]
    mask = torch.reshape(mask, new_shape)
    return x * mask


def randomly_swap_tensor(x, p, width=1, axis=0, limit_range=None):
    """
    Swap random tensor elements along a specified axis

    See `swap_tensor`, except with random `idx_src` and `idx_dst`

    Parameters
    ----------
    x : torch.Tensor<...>
        Input tensor
    p : float
        Fraction of values to swap
    width: int
        Number of contiguous value to swap
    axis: int
        Tensor axis to swap along
    limit_range: tuple(int, int) or None
        Range allowed to be swapped. None means:
        limit_range=x.shape[axis]-width (default is None)

    Returns
    -------
    torch.Tensor<...>
    """
    # Grab tensor dimensions
    shape = x.shape
    # Generate source and destination start indices
    if limit_range is None:
        length = shape[axis]
        n = length - width
    else:
        n = limit_range.copy()
        n[1] = n[1] - width
        length = n[1] - n[0]
    ratio = length / width
    num_starts = int(p * ratio)
    starts_src = sample_n_choose_k(n=n , k=num_starts)
    starts_dst = sample_n_choose_k(n=n, k=num_starts)
    # Add extra, contiguous indices from each start
    contiguous = torch.arange(width)
    idx_src = torch.reshape(starts_src[:, None] + contiguous[None], (-1,))
    idx_dst = torch.reshape(starts_dst[:, None] + contiguous[None], (-1,))
    # Perform swap src->dst for elements along the specified axis
    return swap_tensor(x, idx_src=idx_src, idx_dst=idx_dst, axis=axis)


def swap_tensor(x, idx_src, idx_dst, axis=0):
    """
    Swaps tensor elements along a specified axis

    Example:
        x = [[0, 1, 2, 3, 4],  # Given this tensor
             [5, 6, 7, 8, 9]]
        idx_src = [0, 2]       # Copy positions (0, 2)
        idx_dst = [4, 0]       # Into positions (4, 2)
        axis = 1               # Along second axis (columns)
    Will result in
        y = [[2, 1, 2, 3, 0],
             [7, 6, 7, 8, 1]]

    Note that this is not a permutation, as column 2
    is duplicated and column 5 is completely lost in
    the example above.

    If multiple "src" indices point to the same "dst" index,
    then the contents of the last conflicting "src" index will
    be present into the end result.

    Parameters
    ----------
    x : torch.Tensor<...>
        Input tensor
    idx_src : torch.Tensor<length>
        Indices of values to copy (along given axis)
    idx_dst : torch.Tensor<length>
        Indices of location to paste value (along given axis)
    axis : int
        Axis to apply swapping

    Returns
    -------
    torch.Tensor<...>
    torch.Tensor<length>
    """
    # Transpose x so that the target axis is at position 0
    x = swap_axes(x, axis_a=0, axis_b=axis)
    x[idx_dst] = x[idx_src]
    # Transpose the swapped axis back to its original position
    x = swap_axes(x, axis_a=axis, axis_b=0)
    # Compute mask that is zero on all dst indices
    mask = gen_contiguous_mask(length=x.shape[axis], starts=idx_dst)
    return x, mask


def swap_axes(x, axis_a, axis_b):
    """Transposes a tensor so that two axes are swapped (np.swapaxes)

    Parameters
    ----------
    x : torch.Tensor<...>
        Tensor to swap on (swap will not h)
    axis_a : int
        Tensor axis to swap from/to
    axis_b : int
        Tensor axis to swap to/form

    Returns
    -------
    torch.Tensor<...>
        The input tensor, except with axis_a and axis_b swapped
    """
    transpose = list(range(len(x.shape)))
    transpose[axis_a], transpose[axis_b] = transpose[axis_b], transpose[axis_a]
    return torch.permute(x, transpose)



def get_num_slices(
    total_amount,
    sequence_length,
    frame_shift=1,
    padding=False
):
    '''Number of windows with frame shift in sliding window

    Parameters
    ----------
    total_amount (int)
    sequence_length (int)
    frame_shift (int), optional
    padding (bool), optional:
        If total_amount cannot be split perfectly into equaly sized
        windows, shall the last window be removed (keep_last=False)
        or not (keep_last=True)? (Default: False)

    Returns
    -------
    (int): Number of slices a tensor of length total_amount can be
        divided given sequence_length and frame_shift

    '''
    if padding:
        return max(1, int(np.ceil(total_amount/frame_shift)))
    else:
        return max(1, int(np.ceil((total_amount+1-sequence_length)/frame_shift)))


def unfold_windows(arr, window_size, window_shift,
                   overlap_kind='mean'):
    '''

    Parameters
    ----------
    arr: np.array
        Either 2 or 3 dimensional
    window_size: int
    window_shift: int
    overlap_kind: str, optional
        What to do with possible overlapping areas. (default is 'sum')
        'sum' adds the values in the overlapping areas
        'mean' computes the mean of the overlapping areas

    Returns
    -------
    : np.arr
        2-dimensional array

    '''
    nseg = arr.shape[0]
    last_dim = arr.shape[-1]
    new_dim = (window_shift * nseg + window_size - window_shift, last_dim)
    buffer = np.zeros(new_dim)
    if overlap_kind == 'sum':
        for i in range(nseg):
            buffer[i*window_shift:i*window_shift+window_size] += arr[i]
        return buffer
    elif overlap_kind == 'mean':
        weights = np.zeros((new_dim[0],1))
        for i in range(nseg):
            buffer[i*window_shift:i*window_shift+window_size] += arr[i]
            weights[i*window_shift:i*window_shift+window_size] += 1.0
        return buffer/weights
    else:
        raise NotImplementedError(f'overlap_kind {overlap_kind}')


def windowed_labels(
    labels,
    num_labels,
    frame_length,
    frame_step=None,
    pad_end=False,
    kind='density',
):
    """Generates labels that correspond to STFTs

    With kind=None we are able to split the given labels
    array into batches. (T, C) -> (B, T', C)

    Parameters
    ----------
    labels : np.array

    Returns
    -------
    np.array
    """
    # Labels should be a single vector (int-likes) or kind has to be None
    labels = np.asarray(labels)
    if kind is not None and not labels.ndim == 1:
        raise ValueError('Labels must be a vector')
    if not (labels >= 0).all():
        raise ValueError('All labels must be >= 0')
    if not (labels < num_labels).all():
        raise ValueError(f'All labels must be < {num_labels} (num_labels)')
    # Kind determines how labels in each window should be processed
    if not kind in {'counts', 'density', 'onehot', 'argmax', None}:
        raise ValueError('`kind` must be in {counts, density, onehot, argmax, None}')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process labels with a sliding window. TODO: vectorize?
    output = []
    for i in range(0, len(labels), frame_step):
        chunk = labels[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        # Just append the chunk if kind is None
        if kind == None:
            output.append(chunk)
            continue
        # Count the occurences of each label
        counts = np.bincount(chunk, minlength=num_labels)
        # Then process based on kind
        if kind == 'counts':
            output.append(counts)
        elif kind == 'density':
            output.append(counts / len(chunk))
        elif kind == 'onehot':
            one_hot = np.zeros(num_labels)
            one_hot[np.argmax(counts)] = 1
            output.append(one_hot)
        elif kind == 'argmax':
            output.append(np.argmax(counts))
    return np.array(output)


def windowed_signals(
    signals,
    frame_length,
    frame_step=None,
    pad_end=False
):
    """Generates signal segments of size frame_length"""
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process signals with a sliding window
    output = []
    for i in range(0, len(signals), frame_step):
        chunk = signals[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        output.append(chunk)
    return np.array(output)


EPS=1e-10
def normalize(x, mean, std):
    '''Normalizes the given tensor with Standard scaler'''
    return (x - mean) / std

def rev_normalize(x, mean, std):
    '''Undo normalization with Standard scaler'''
    return x*std + mean


def normalize_min_max(x, _min, _max):
    '''Normalizes given tensor with MinMax scaler'''
    return (x-_min)/(_max-_min)


def meter_per_sec_squared2g(x):
    '''Transforms acceleration in m/s^2 to g unit'''
    return x/9.80665
