import pandas as pd
import numpy as np
import random 
import constants
from typing import Any, Callable, List, Optional, Tuple, Union
from numpy.typing import NDArray
import tqdm
import os
import mne


from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import torch
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from constants import INCLUDED_CHANNELS


def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform

    # print("in size: ",signals.shape)
    # print("before fft: ",signals.shape)

    # print("clip: ",clip.shape)
    fourier_signal = fft(signals, n=n*2, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0
    # print("out size: ",amp.shape)

    FT = np.log(amp)
    # print("ft: ",FT.shape)
    P = np.angle(fourier_signal)


    return FT, P


def butter_bandpass( lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# 批量应用带通滤波器
def batch_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # 在时间维度应用滤波器
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

class signal_transform:
    def __init__(self,args,scaler) -> None:
        self.use_fft = args.use_fft
        # self.mean = mean
        # self.std = std
        self.scaler = scaler
        
    def transform(self,data):
        if(self.use_fft):
            data, _ = computeFFT(
            data, n=constants.FREQUENCY)
        #     print(dt.shape)
        # scaler = StandardScaler(mean=self.mean,std=self.std)
        # scaler.fit(data)
        L,C = data.shape
        
        data_channel = data.reshape(-1,1)

        whole_data = self.scaler.transform(data_channel)
        data_transformed = whole_data.reshape(L,C)

        return data_transformed

def binary_label_transform(data):
    df = data
    df['label'] = data['label'].apply(lambda x: 1 if x > 1 else x)
    return df


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

    def forward(self, data: np.ndarray):
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




def calc_data_scale(args):

    scaler = StandardScaler()
    if args.dataset =="BSI":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed.npy')
    elif args.dataset =="BSI1s":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_1s.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_1s.npy')

    elif args.dataset =="BSI2s":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_2s.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_2s.npy')
    elif args.dataset == "BSI5class":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_5class.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_5class.npy')
    
    elif args.dataset == "BSIaligned":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_aligned1_2.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_aligned1_2.npy')

    elif args.dataset == "BSIunaligned":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_unaligneds1_2.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_unaligneds1_2.npy')

    elif args.dataset == "BSIalignedus":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_alignedus1_2.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_alignedus1_2.npy')
    
    elif args.dataset == "BSIstreamaligned":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_streamaligned.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_streamaligned.npy')

    elif args.dataset == "BSI+onset":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train+onset.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train+onset.npy')

    elif args.dataset == "BSI+onset+una":
        x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train+onset+una.npy')
        y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train+onset+una.npy')

    elif args.dataset == "BSIsample":
        x = np.load(args.dataset_dir + 'X_train_samplew1s01.npy')
        y = np.load(args.dataset_dir + 'label_train_samplew1s01.npy')

    elif args.dataset == "BSIsamplewavelet":
        x = np.load(args.dataset_dir + 'X_train_samplewavelet.npy')
        y = np.load(args.dataset_dir + 'label_train_samplewavelet.npy')


    _, counts = np.unique(y, return_counts=True)

    if args.use_fft:
        x = x.transpose(0,2,1)
        # print("fit: ",x.shape)
        fft_result = np.fft.fft(x, axis=-1)
        abs_result = np.abs(fft_result[:, :, :x.shape[1] // 2])
        xfft = np.log(abs_result + 1e-8)  # 防止对零取对数

        scaler.fit(xfft.reshape(xfft.shape[0],-1))
        print("fft mean: ",scaler.mean_)
        print("fft var: ",scaler.var_)

    elif args.preprocess:
        filter = MNEFilter(sfreq=590, l_freq=1, h_freq=200, notch_freqs=np.arange(50, 201, 50), apply_car=True)

        N,T,D = x.shape
        x = x.reshape((T,-1))
        x = filter.forward(x)

        x = x.reshape(N,-1)
        scaler.fit(x)

        print("process mean: ",scaler.mean_)
        print("process var: ",scaler.var_)
    
    else:
        # x_channel = x - np.mean(x,axis=2,keepdims = True)
        x_channel = x
        x_channel = x_channel.reshape(x.shape[0],-1)
        scaler.fit(x_channel)


        print("mean: ",scaler.mean_)
        print("var: ",scaler.var_)

    # means = np.mean(x,axis=2)
    # stds = np.std(x,axis=2)
    return scaler,  counts

# def read_sessions(file_dir):

#     file_marker_dir = os.path.join(file_dir)
#     with open(file_marker_dir, 'r') as f:
#         f_str = f.readlines()
    
#     sessions = []
#     for i in range(len(f_str)):
#         tup = f_str[i].strip("\n").split(",")
#         tup[1] = int(tup[1]) # seizure class
#         tup[2] = int(tup[2]) # seizure index
#         session_name = tup[0].split(".")
#         session_name = session_name[0]
#         sessions.append(session_name)
#     #     file_tuples.append(tup)
#     # size = len(file_tuples)
#     return sessions

# def read_sessions_detect(file_dir):
#     current_path = os.getcwd()
#     print("current: ",current_path)
#     file_marker_dir = os.path.join(current_path+file_dir)
#     with open(file_marker_dir, 'r') as f:
#         f_str = f.readlines()
    
#     sessions = []
#     for i in range(len(f_str)):
#         tup = f_str[i].strip("\n").split(",")
#         tup[1] = int(tup[1]) # seizure class
#         session_name = tup[0].split(".")
#         session_name = session_name[0]
#         sessions.append(session_name)
#     #     file_tuples.append(tup)
#     # size = len(file_tuples)
#     return sessions

# def read_sessions_ssl(file_dir):

#     file_marker_dir = os.path.join(file_dir)
#     with open(file_marker_dir, 'r') as f:
#         f_str = f.readlines()
    
#     sessions = []
#     session_names = []
#     for i in range(len(f_str)):
#         tup = f_str[i].strip("\n").split(",")
#         x,y = tup
#         x_id = int(x.split('_')[-1].split('.h5')[0])
#         y_id = int(y.split('_')[-1].split('.h5')[0])
#         session_name = x.split('.edf')[0]
#         sessions.append((session_name,x_id,y_id))
#         session_names.append(session_name)
#     #     file_tuples.append(tup)
#     # size = len(file_tuples)
#     return sessions,session_names

# def search_intersect(la,sr):
#     list_set = set(la)
#     # print(list_set[:10])
#     column_set = set(sr)
#     # print(column_set[:10])
#     intersection = list_set.intersection(column_set)
#     return list(intersection)

# def train_valid_data_selection(args,task_op=None):
#     # print("task op: ",task_op)
#     if task_op is not None:
#         TASK = task_op
#     else:
#         TASK = args.task
#     print("current task: ",TASK)
#     train_df = None
#     valid_df = None

#     # if args.task=="Detection":
#     #     path1 = "src/data/detection_data/train_x.npy"
#     #     path2 = "src/data/detection_data/train_y.npy"
#     #     path3 = "src/data/detection_data/valid_x.npy"
#     #     path4 = "src/data/detection_data/valid_y.npy"
#     #     if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3) and os.path.exists(path4):
#     #         print("detection data already exists, skip")
#     #         return train_df,valid_df
#     if TASK=="Detection":
#         path1 = "/SCRATCH2/yuhxie/clips/detection_train.parquet"
#         path2 = "/SCRATCH2/yuhxie/clips/detection_valid.parquet"
#         if os.path.exists(path1) and os.path.exists(path2):
#             print("detection train valid clips already exists, skip")
#             train_df = pd.read_parquet(path1)
#             valid_df = pd.read_parquet(path2)
#             return train_df,valid_df
#     if TASK=="SSL":
#         current_path = os.getcwd()
#         path1 = "/SCRATCH2/yuhxie/clips/ssl_trainx.parquet"
#         path2 = "/SCRATCH2/yuhxie/clips/ssl_trainy.parquet"
#         path3 = "/SCRATCH2/yuhxie/clips/ssl_validx.parquet"
#         path4 = "/SCRATCH2/yuhxie/clips/ssl_validy.parquet"
#         if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3) and os.path.exists(path4):
#             print("ssl clips already exists, skip")
#             train_dfx = pd.read_parquet(path1)
#             train_dfy = pd.read_parquet(path2)
#             valid_dfx = pd.read_parquet(path3)  
#             valid_dfy = pd.read_parquet(path4)
#             return train_dfx,train_dfy,valid_dfx,valid_dfy

#     if TASK=="SSLDetection":
#         path1 = "/SCRATCH2/yuhxie/clips/ssl_detection_trainx.parquet"
#         path2 = "/SCRATCH2/yuhxie/clips/ssl_detection_trainy.parquet"
#         path3 = "/SCRATCH2/yuhxie/clips/ssl_detection_validx.parquet"
#         path4 = "/SCRATCH2/yuhxie/clips/ssl_detection_validy.parquet"
#         if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3) and os.path.exists(path4):
#             print("ssl clips already exists, skip")
#             train_dfx = pd.read_parquet(path1)
#             train_dfy = pd.read_parquet(path2)
#             valid_dfx = pd.read_parquet(path3)  
#             valid_dfy = pd.read_parquet(path4)
#             return train_dfx,train_dfy,valid_dfx,valid_dfy

#     if TASK=="Classification":
#         train_sessions = read_sessions("/SCRATCH2/yuhxie/file_markers_classification/trainSet_seizure_files.txt")
#         valid_sessions = read_sessions("/SCRATCH2/yuhxie/file_markers_classification/devSet_seizure_files.txt")
#         event_df = (
#         pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
        
#         .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=("pre-ictal",0), overlap_action='seizure', sort_index=True)
#         .pipe(segments_by_labels, target_labels=[1,2,3,4], relabel=True)  
#         )
#         event_df["start_time"] = event_df["start_time"] -args.pre_ictal_length

#         train_inter = search_intersect(train_sessions,list(event_df.index.get_level_values('session')))
#         valid_inter = search_intersect(valid_sessions,list(event_df.index.get_level_values('session')))


#         train_df = event_df.loc[pd.IndexSlice[:, train_inter, :, :]]
#         valid_df = event_df.loc[pd.IndexSlice[:, valid_inter, :, :]]
#         train_df = train_df.sort_index()
#         valid_df = valid_df.sort_index()


#     elif TASK=="Detection":

#         print("detection")
#         train_sessions_nosz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/trainSet_seq2seq_12s_nosz.txt")
#         train_sessions_sz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/trainSet_seq2seq_12s_sz.txt")
#         valid_sessions_nosz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/devSet_seq2seq_12s_nosz.txt")
#         valid_sessions_sz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/devSet_seq2seq_12s_sz.txt")
#         train_sessions = train_sessions_nosz + train_sessions_sz
#         valid_sessions = valid_sessions_nosz + valid_sessions_sz
#         whole_sessions = train_sessions + valid_sessions
#         df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
#         df_train_ns = df.loc[pd.IndexSlice[:, train_sessions_nosz, :, :]]
#         df_train_s = df.loc[pd.IndexSlice[:, train_sessions_sz, :, :]]
#         df_vali = df.loc[pd.IndexSlice[:, valid_sessions, :, :]]


#         # df = df.loc[pd.IndexSlice[:, whole_sessions, :, :]]

#         train_df_ns = (
#         df_train_ns
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=False)
#         .pipe(extract_target_labels, target_labels=[0], relabel=True)
#         )
#         train_df_s = (
#         df_train_s
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=False)
#         .pipe(extract_target_labels, target_labels=[1,2,3,4], relabel=False)
#         )
#         trainsz_inter = search_intersect(train_sessions_sz,list(train_df_s.index.get_level_values('session')))
#         train_df_s = train_df_s.loc[pd.IndexSlice[:, trainsz_inter, :, :]]
#         seizures = len(train_df_s)
#         print("seizure length: ",seizures)
#         train_df_ns = train_df_ns.sample(n=seizures)

#         # train_df_ns = train_df_ns[:seizures]
#         print("df_train_ns length: ",len(train_df_ns))
#         train_df = pd.concat([train_df_ns,train_df_s])
#         # train_df.sort_index()
#         # patients = train_df.groupby('patient')
#         # print("count of patients: ",patients.count())

#         labels1 = train_df_ns.groupby('label')
#         print("count of labels ns: ",labels1.count())

#         labels2 = train_df_s.groupby('label')
#         print("count of labels s: ",labels2.count())

#         labels3 = train_df.groupby('label')
#         print("count of labels: ",labels3.count())



#         # event_df = event_df[event_df.index.get_level_values('segment') == 0]
#         valid_df = (
#         df_vali
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )
#         train_df = binary_label_transform(train_df)
#         valid_df = binary_label_transform(valid_df)
#         # labels = train_df.groupby('label')
#         # print("label class: ",labels.count())
        
#         # train_inter = search_intersect(train_sessions,list(event_df.index.get_level_values('session')))
#         # valid_inter = search_intersect(valid_sessions,list(event_df.index.get_level_values('session')))
#         print("saving detection train valid")
#         current_path = os.getcwd()
#         group = train_df.groupby("session")
#         train_list = []
#         for name,seg in group:
#             # print("group name",name,len(seg))
#             seg = seg.sort_values(by="segment")
#             # print(seg[:5])
#             train_list.append(seg)
#         train_df = pd.concat(train_list,axis=0)
#         # train_df.sort_index(by = "session").groupby('session').apply(lambda x: x.sort_values()).reset_index(drop=True)
#         # train_df.sort_index(level = "session")
#         valid_df.sort_index()
#         print("train_df sort: ")
#         print(train_df[:15])
        
#         train_df.to_parquet("/SCRATCH2/yuhxie/clips/detection_train.parquet")
#         valid_df.to_parquet("/SCRATCH2/yuhxie/clips/detection_valid.parquet")
        

#         # valid_df = event_df.loc[pd.IndexSlice[:, valid_inter, :, :]]
#     elif TASK=="SSL":

#         # print("detection data already exists, skip")
#         # return train_df,valid_df

#         print("ssl")
#         train_sessions,train_session_names = read_sessions_ssl("/SCRATCH2/yuhxie/file_markers_ssl/trainSet_seq2seq_12s.txt")
#         valid_sessions,valid_session_names = read_sessions_ssl("/SCRATCH2/yuhxie/file_markers_ssl/devSet_seq2seq_12s.txt")
#         df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
#         # df_train = df.loc[pd.IndexSlice[:, train_session_names, :, :]]
#         # df_vali = df.loc[pd.IndexSlice[:, valid_session_names, :, :]]

#         print("ssl train valid num: ",len(train_sessions),len(valid_sessions))
#         # df = df.loc[pd.IndexSlice[:, whole_sessions, :, :]]

#         train_df = (
#         df
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )
#         train_inter = search_intersect(train_session_names,list(train_df.index.get_level_values('session')))
#         new_train_sessions = [t for t in train_sessions if t[0] in train_inter]

#         valid_df = (
#         df
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )
#         valid_inter = search_intersect(valid_session_names,list(valid_df.index.get_level_values('session')))
#         new_valid_sessions = [t for t in valid_sessions if t[0] in valid_inter]

#         df_train = train_df.loc[pd.IndexSlice[:, train_inter, :, :]]
#         df_vali = valid_df.loc[pd.IndexSlice[:, valid_inter, :, :]]

#         print("inter: ",df_train.shape,train_df.shape,df_vali.shape,valid_df.shape)

#         train_session = df_train.groupby('session')
#         valid_session = df_vali.groupby('session')

#         train_listx = []
#         train_listy = []
#         valid_listx = []
#         valid_listy = []

#         # print(len(train_session))
#         for name,session in train_session:
#             # print("session length",session.shape)
#             session_x = session[:-1]
#             session_y = session[1:]
#             # print("x: ",session.shape,session_x.shape,session_y.shape)
#             train_listx.append(session_x)
#             train_listy.append(session_y)
#         train_dfx = pd.concat(train_listx,axis=0)
#         train_dfy = pd.concat(train_listy,axis=0)

#         for name,session in valid_session:
#             # print("session length",session.shape)
#             session_x = session[:-1]
#             session_y = session[1:]
#             # print("x: ",session.shape,session_x.shape,session_y.shape)
#             valid_listx.append(session_x)
#             valid_listy.append(session_y)
#         valid_dfx = pd.concat(valid_listx,axis=0)
#         valid_dfy = pd.concat(valid_listy,axis=0)

#         # print(train_dfx[:50])
#         # print(train_dfy[:50])


#         print("ssl train valid num: ",len(new_train_sessions),len(new_valid_sessions))
#         # return new_train_sessions, new_valid_sessions
#         # current_path = os.getcwd()
#         # train_df.to_parquet(current_path+"/src/data/clips/ssl_train.parquet")
#         # valid_df.to_parquet(current_path+"/src/data/clips/ssl_valid.parquet")

#         current_path = os.getcwd()
#         train_dfx.to_parquet("/SCRATCH2/yuhxie/clips/ssl_trainx.parquet")
#         train_dfy.to_parquet("/SCRATCH2/yuhxie/clips/ssl_trainy.parquet")
#         valid_dfx.to_parquet("/SCRATCH2/yuhxie/clips/ssl_validx.parquet")
#         valid_dfy.to_parquet("/SCRATCH2/yuhxie/clips/ssl_validy.parquet")
#         return train_dfx,train_dfy,valid_dfx,valid_dfy

#     elif TASK=="SSLDetection":

#         print("ssl_detection")
#         train_sessions_nosz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/trainSet_seq2seq_12s_nosz.txt")
#         train_sessions_sz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/trainSet_seq2seq_12s_sz.txt")
#         valid_sessions_nosz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/devSet_seq2seq_12s_nosz.txt")
#         valid_sessions_sz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/devSet_seq2seq_12s_sz.txt")
#         train_sessions = train_sessions_nosz + train_sessions_sz
#         valid_sessions = valid_sessions_nosz + valid_sessions_sz
#         df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/train/segments.parquet")
#         df_train_ns = df.loc[pd.IndexSlice[:, train_sessions_nosz, :, :]]
#         df_train_s = df.loc[pd.IndexSlice[:, train_sessions_sz, :, :]]
#         df_vali = df.loc[pd.IndexSlice[:, valid_sessions, :, :]]


#         # df = df.loc[pd.IndexSlice[:, whole_sessions, :, :]]

#         train_df_ns = (
#         df_train_ns
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=False)
#         .pipe(extract_target_labels, target_labels=[0], relabel=True)
#         )
#         train_df_s = (
#         df_train_s
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=False)
#         .pipe(extract_target_labels, target_labels=[1,2,3,4], relabel=False)
#         )
#         trainsz_inter = search_intersect(train_sessions_sz,list(train_df_s.index.get_level_values('session')))
#         train_df_s = train_df_s.loc[pd.IndexSlice[:, trainsz_inter, :, :]]
#         seizures = len(train_df_s)
#         print("seizure length: ",seizures)
#         train_df_ns = train_df_ns.sample(n=seizures)

#         # train_df_ns = train_df_ns[:seizures]
#         print("df_train_ns length: ",len(train_df_ns))
#         train_df = pd.concat([train_df_ns,train_df_s])
#         # train_df.sort_index()
#         # patients = train_df.groupby('patient')
#         # print("count of patients: ",patients.count())

#         labels1 = train_df_ns.groupby('label')
#         print("count of labels ns: ",labels1.count())

#         labels2 = train_df_s.groupby('label')
#         print("count of labels s: ",labels2.count())

#         labels3 = train_df.groupby('label')
#         print("count of labels: ",labels3.count())



#         # event_df = event_df[event_df.index.get_level_values('segment') == 0]
#         valid_df = (
#         df_vali
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )

#         train_session = train_df.groupby('session')
#         valid_session = valid_df.groupby('session')

#         train_listx = []
#         train_listy = []
#         valid_listx = []
#         valid_listy = []

#         # print(len(train_session))
#         for name,session in train_session:
#             # print("session length",session.shape)
#             session_x = session[:-1]
#             session_y = session[1:]
#             # print("x: ",session.shape,session_x.shape,session_y.shape)
#             train_listx.append(session_x)
#             train_listy.append(session_y)
#         train_dfx = pd.concat(train_listx,axis=0)
#         train_dfy = pd.concat(train_listy,axis=0)

#         for name,session in valid_session:
#             # print("session length",session.shape)
#             session_x = session[:-1]
#             session_y = session[1:]
#             # print("x: ",session.shape,session_x.shape,session_y.shape)
#             valid_listx.append(session_x)
#             valid_listy.append(session_y)
#         valid_dfx = pd.concat(valid_listx,axis=0)
#         valid_dfy = pd.concat(valid_listy,axis=0)

#         # print("ssldetection train valid num: ",len(new_train_sessions),len(new_valid_sessions))

#         current_path = os.getcwd()
#         train_dfx.to_parquet("/SCRATCH2/yuhxie/clips/ssl_detection_trainx.parquet")
#         train_dfy.to_parquet("/SCRATCH2/yuhxie/clips/ssl_detection_trainy.parquet")
#         valid_dfx.to_parquet("/SCRATCH2/yuhxie/clips/ssl_detection_validx.parquet")
#         valid_dfy.to_parquet("/SCRATCH2/yuhxie/clips/ssl_detection_validy.parquet")
#         return train_dfx,train_dfy,valid_dfx,valid_dfy


        


#     print("train valid: ",train_df.shape,valid_df.shape)

         
#     # grouped = event_df.groupby('session')
#     # all_sessions = list(grouped.groups.keys())
#     # # set the sample rate
#     # groupsize = int(len(grouped))

#     # # sampled_sessions = random.sample(all_sessions, sample_size)

#     # # randomly choose selected sessions
#     # train_size = int(groupsize * 0.75)

#     # train_sessions = all_sessions[:train_size]
#     # valid_sessions = all_sessions[train_size:]
    


#     # # divide train and test dataset
#     # train_df = pd.concat([grouped.get_group(session) for session in train_sessions])
#     # valid_df = pd.concat([grouped.get_group(session) for session in valid_sessions])


#     return train_df,valid_df
    


# def test_data_selection(args,task_op = None):
#     if task_op is not None:
#         TASK = task_op
#     else:
#         TASK = args.task
#     print("current task: ",TASK)
#     test_df = None

#     # if args.task == "Detection":
#     #     path1 = "src/data/detection_data/test_x.npy"
#     #     path2 = "src/data/detection_data/test_y.npy"
#     #     if os.path.exists(path1) and os.path.exists(path2):
#     #         print("detection data already exists, skip")
#     #         return test_df
#     if TASK == "Detection":
#         current_path = os.getcwd()
#         path1 = "/SCRATCH2/yuhxie/clips/detection_test.parquet"
#         if os.path.exists(path1):
#             print("detection test clips already exists, skip")
#             test_df = pd.read_parquet(path1)
#             return test_df
        
#     if TASK == "SSL":
#         current_path = os.getcwd()
#         path1 = "/SCRATCH2/yuhxie/clips/ssl_testx.parquet"
#         path2 = "/SCRATCH2/yuhxie/clips/ssl_testy.parquet"
#         if os.path.exists(path1) or os.path.exists(path2):
#             print("ssl clips already exists, skip")
#             test_dfx = pd.read_parquet(path1)
#             test_dfy = pd.read_parquet(path2)
#             return test_dfx,test_dfy
    
#     if TASK == "SSLDetection":
#         current_path = os.getcwd()
#         path1 = "/SCRATCH2/yuhxie/clips/ssl_detection_testx.parquet"
#         path2 = "/SCRATCH2/yuhxie/clips/ssl_detection_testy.parquet"
#         if os.path.exists(path1) or os.path.exists(path2):
#             print("ssl clips already exists, skip")
#             test_dfx = pd.read_parquet(path1)
#             test_dfy = pd.read_parquet(path2)
#             return test_dfx,test_dfy
    
#     if TASK=="Classification":
#         test_sessions = read_sessions("/SCRATCH2/yuhxie/file_markers_classification/testSet_seizure_files.txt")
#         test_df = (
#         pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/dev/segments.parquet")
        
#         .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=("pre-ictal",0), overlap_action='seizure', sort_index=True)
#         .pipe(segments_by_labels, target_labels=[1,2,3,4], relabel=True)
        
#         )
#         test_df["start_time"] = test_df["start_time"] -args.pre_ictal_length
#         # print(list(test_df.index.get_level_values('session')))
#         test_inter = search_intersect(test_sessions,list(test_df.index.get_level_values('session')))
#         test_df = test_df.loc[pd.IndexSlice[:, test_inter, :, :]]
#         test_df = test_df.sort_index()

#     elif TASK=="Detection":

#         print("detection")
#         test_sessions_nosz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/testSet_seq2seq_12s_nosz.txt")
#         test_sessions_sz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/testSet_seq2seq_12s_sz.txt")
#         test_sessions = test_sessions_nosz + test_sessions_sz
#         df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/dev/segments.parquet")

#         test_df = (
#         df
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )

#         test_df = binary_label_transform(test_df)
#         print("saving detection test")
#         current_path = os.getcwd()
#         test_df.sort_index()
#         test_df.to_parquet("/SCRATCH2/yuhxie/clips/detection_test.parquet")
        
#         # valid_df = event_df.loc[pd.IndexSlice[:, valid_inter, :, :]]
#     elif TASK =="SSL":
#         print("ssl")
#         # print("detection data already exists, skip")
#         # return test_df
        
#         test_sessions,test_session_names = read_sessions_ssl("/SCRATCH2/yuhxie/file_markers_ssl/testSet_seq2seq_12s.txt")
#         print("ssl test num: ",len(test_sessions))
#         df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/dev/segments.parquet")

#         test_df = (
#         df
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )
#         test_inter = search_intersect(test_session_names,list(test_df.index.get_level_values('session')))
#         new_test_sessions = [t for t in test_sessions if t[0] in test_inter]
#         df_test = test_df.loc[pd.IndexSlice[:, test_inter, :, :]]
#         print("inter: ",df_test.shape,test_df.shape)
#         # print("ssl test num: ",len(new_test_sessions))

#         test_session = df_test.groupby('session')

#         test_listx = []
#         test_listy = []

#         # print(len(train_session))
#         for name,session in test_session:
#             # print("session length",session.shape)
#             session_x = session[:-1]
#             session_y = session[1:]
#             # print("x: ",session.shape,session_x.shape,session_y.shape)
#             test_listx.append(session_x)
#             test_listy.append(session_y)
#         test_dfx = pd.concat(test_listx,axis=0)
#         test_dfy = pd.concat(test_listy,axis=0)
        

#         current_path = os.getcwd()
#         test_dfx.to_parquet("/SCRATCH2/yuhxie/clips/ssl_testx.parquet")
#         test_dfy.to_parquet("/SCRATCH2/yuhxie/clips/ssl_testy.parquet")

#         return test_dfx,test_dfy

#     elif TASK=="SSLDetection":

#         print("SSL_detection")
#         test_sessions_nosz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/testSet_seq2seq_12s_nosz.txt")
#         test_sessions_sz = read_sessions_detect("/SCRATCH2/yuhxie/file_markers_detection/testSet_seq2seq_12s_sz.txt")
#         test_sessions = test_sessions_nosz + test_sessions_sz
#         df = pd.read_parquet("/datasets2/epilepsy/TUSZ/processed/dev/segments.parquet")

#         test_df = (
#         df
#         # .pipe(patients_by_seizures, low=args.low_seizure, high=args.high_seizure)
#         .pipe(make_clips, clip_length=args.clip_length, clip_stride=args.clip_stride, overlap_action='seizure', sort_index=True)
#         )

#         test_df = binary_label_transform(test_df)
#         print("saving detection test")
#         current_path = os.getcwd()
#         test_df.sort_index()
#         # test_df.to_parquet(current_path+"/src/data/clips/detection_test.parquet")

#         test_session = test_df.groupby('session')

#         test_listx = []
#         test_listy = []

#         # print(len(train_session))
#         for name,session in test_session:
#             # print("session length",session.shape)
#             session_x = session[:-1]
#             session_y = session[1:]
#             # print("x: ",session.shape,session_x.shape,session_y.shape)
#             test_listx.append(session_x)
#             test_listy.append(session_y)
#         test_dfx = pd.concat(test_listx,axis=0)
#         test_dfy = pd.concat(test_listy,axis=0)
        

#         current_path = os.getcwd()
#         test_dfx.to_parquet("/SCRATCH2/yuhxie/clips/ssl_detection_testx.parquet")
#         test_dfy.to_parquet("/SCRATCH2/yuhxie/clips/ssl_detection_testy.parquet")

#         return test_dfx,test_dfy

#         # return new_test_sessions
#         # current_path = os.getcwd()
#         # test_df.to_parquet(current_path+"/src/data/clips/ssl_test.parquet")


    
#     print("test: ",test_df.shape)

#     return test_df

# def count_consecutive_segments(loader,length):
#     grouped = loader.groupby('session')
#     total = 0
#     for name,group in grouped:
#         arr = group["label"].values
#         if arr.shape[0] == 0:
#             return 0
#         if arr.shape[0]==1:
#             total = total+1
#             continue
#         curr_start = 0
#         count = 1  # 初始化计数为1，因为至少有一个段
#         current_element = arr[0]  # 当前元素
#         for i in range(1, arr.shape[0]):
#             if arr[i] != current_element or i-curr_start>length:
#                 count += 1
#                 current_element = arr[i]
#                 curr_start = i
#         total = total+count

#     return total

# def get_swap_pairs(channels):
#     """
#     Swap select adjacenet channels
#     Args:
#         channels: list of channel names
#     Returns:
#         list of tuples, each a pair of channel indices being swapped
#     """
#     swap_pairs = []
#     if ("EEG FP1" in channels) and ("EEG FP2" in channels):
#         swap_pairs.append((channels.index("EEG FP1"), channels.index("EEG FP2")))
#     if ("EEG Fp1" in channels) and ("EEG Fp2" in channels):
#         swap_pairs.append((channels.index("EEG Fp1"), channels.index("EEG Fp2")))
#     if ("EEG F3" in channels) and ("EEG F4" in channels):
#         swap_pairs.append((channels.index("EEG F3"), channels.index("EEG F4")))
#     if ("EEG F7" in channels) and ("EEG F8" in channels):
#         swap_pairs.append((channels.index("EEG F7"), channels.index("EEG F8")))
#     if ("EEG C3" in channels) and ("EEG C4" in channels):
#         swap_pairs.append((channels.index("EEG C3"), channels.index("EEG C4")))
#     if ("EEG T3" in channels) and ("EEG T4" in channels):
#         swap_pairs.append((channels.index("EEG T3"), channels.index("EEG T4")))
#     if ("EEG T5" in channels) and ("EEG T6" in channels):
#         swap_pairs.append((channels.index("EEG T5"), channels.index("EEG T6")))
#     if ("EEG O1" in channels) and ("EEG O2" in channels):
#         swap_pairs.append((channels.index("EEG O1"), channels.index("EEG O2")))

#     return swap_pairs

# def _random_reflect(EEG_seq):
#     """
#     Randomly reflect EEG channels along the midline
#     """
#     swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
#     EEG_seq_reflect = EEG_seq.copy()
#     if(np.random.choice([True, False])):

#         for pair in swap_pairs:
#             EEG_seq_reflect[:, :,[pair[0], pair[1]]] = EEG_seq[:, :, [pair[1], pair[0]]]
#     else:

#         swap_pairs = None
#     return EEG_seq_reflect, swap_pairs

# def _random_scale(fft,EEG_seq):
#     """
#     Scale EEG signals by a random value between 0.8 and 1.2
#     """
    
#     scale_factor = np.random.uniform(0.8, 1.2)
#     if fft:
#         # print("fft version!")
#         # print("channel mean: ",np.mean(EEG_seq,axis=0),np.max(EEG_seq,axis=0),np.min(EEG_seq,axis=0))
#         EEG_seq += np.log(scale_factor)
#     else:
#         EEG_seq *= scale_factor
#     return EEG_seq

# def _random_reflect_ssl(EEG_seq1,EEG_seq2):
#     """
#     Randomly reflect EEG channels along the midline
#     """
#     swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
#     EEG_seq_reflect1 = EEG_seq1.copy()
#     EEG_seq_reflect2 = EEG_seq2.copy()
#     if(np.random.choice([True, False])):

#         for pair in swap_pairs:
#             EEG_seq_reflect1[:, :,[pair[0], pair[1]]] = EEG_seq1[:, :, [pair[1], pair[0]]]
#             EEG_seq_reflect2[:, :,[pair[0], pair[1]]] = EEG_seq2[:, :, [pair[1], pair[0]]]
#     else:

#         swap_pairs = None
#     return EEG_seq_reflect1,EEG_seq_reflect2, swap_pairs

# def _random_scale_ssl(fft,EEG_seq1,EEG_seq2):
#     """
#     Scale EEG signals by a random value between 0.8 and 1.2
#     """
    
#     scale_factor = np.random.uniform(0.8, 1.2)
#     if fft:
#         # print("fft version!")
#         # print("channel mean: ",np.mean(EEG_seq,axis=0),np.max(EEG_seq,axis=0),np.min(EEG_seq,axis=0))
#         EEG_seq1 += np.log(scale_factor)
#         EEG_seq2 += np.log(scale_factor)
#     else:
#         EEG_seq1 *= scale_factor
#         EEG_seq2 *= scale_factor
#     return EEG_seq1,EEG_seq2







if __name__ == "__main__":
    # dataloader = test_data_selection()
    for x,y in test_data_selection():
        print(x.shape,y.shape)
    
    

    