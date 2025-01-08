from torch.utils.data import Dataset, DataLoader
from data.data_utils import computeFFT
from torch_geometric.data.dataset import Dataset as PyGDataset
from torch_geometric.data import Data as PyGData
import constants
import scipy.signal as signal
import numpy as np
import pickle
import torch
import time
import utils
import constants
import pandas as pd
import pywt
from scipy.signal import decimate
from sklearn.decomposition import PCA

import numpy as np
from scipy.signal import butter, sosfilt

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    单个带通滤波器函数，对信号的最后一维进行滤波。
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='bp',output='sos')  # 设计带通滤波器
    return sosfilt(sos, data, axis=0)  # 应用滤波

def multi_band_filter(data, bounds, fs, order=5):
    """
    多频带带通滤波器，对每个频带进行滤波并堆叠结果。
    
    参数：
        data (numpy.ndarray): 输入信号，形状为 (B, C, T)。
        bounds (list of tuples): 每个频带的低高截止频率列表，如 [(low1, high1), (low2, high2), ...]。
        fs (float): 采样频率（Hz）。
        order (int): 滤波器的阶数。
        
    返回：
        numpy.ndarray: 滤波后的信号，形状为 (B, len(bounds) * C, T)。
    """
    C, T = data.shape
    filtered_signals = []  # 用于存储每个频带的结果

    for lowcut, highcut in bounds:
        filtered = bandpass_filter(data, lowcut, highcut, fs, order)  # 对当前频带滤波
        filtered_signals.append(np.expand_dims(filtered, axis=0))  # 结果加入列表

    # 沿通道维度堆叠结果
    return np.concatenate(filtered_signals, axis=0)




class Fingerflex_Dataset(Dataset):
    """
    The class that defines the sampling unit
    """
    def __init__(self, x,y , args, split):
        """
        paths should point to .npy files
        """
        self.ecog_data, self.fingerflex_data = x,y
        
        self.duration = self.ecog_data.shape[-1]
        self.sample_len = args.sample_len                                 # sample size
        self.stride = 1                                              # stride between samples
        self.ds_len = (self.duration-self.sample_len) // self.stride
        self.args = args
        
        print("Duration: ", self.duration, "Ds_len:", self.ds_len)
    def __len__(self):
        return self.ds_len
    
    def __getitem__(self, index):

        sample_start = index*self.stride
        sample_end = sample_start+self.sample_len

        ecog_sample = self.ecog_data[...,sample_start:sample_end] # x
        
        channels,_ = ecog_sample.shape

        ecog_sample = ecog_sample.reshape((channels* self.args.num_cut,-1))
        
        fingerflex_sample = self.fingerflex_data[...,sample_start:sample_end] # y
        
        return ecog_sample, fingerflex_sample

    
class BSI_Dataset(Dataset):
    def __init__(self, x,y,args,split,scaler):
        """
        Args:
            data (list): 包含数据样本的列表。
            labels (list): 包含标签的列表，与数据样本一一对应。
            transform (callable, optional): 对数据样本进行预处理的可调用函数。如果不需要预处理，可以设置为None。
        """
        self.x = x
        self.y = y
        self.split = split
        self.args = args
        self.device = args.gpu
        self.scaler = scaler

    def __len__(self):
        return self.y.shape[0]
    
    def adding_noise(self,x):
        noise = np.random.randn(*(x.shape))*self.args.aug_variance
        x = x + noise

        # random_int = np.random.randint(0, int(self.args.mask_rate * x.shape[0])+1)

        # indices = torch.randperm(x.shape[0])[:random_int]

        # x[indices,:] = 0

        return x
    
    def scaling(self,data):

        
        xshape = data.shape
        data_channel = data.reshape(data.shape[0],-1)

        whole_data = self.scaler.transform(data_channel)
        data_transformed = whole_data.reshape(xshape)

        # data_transformed = (data - data.min(axis=(0, 1), keepdims=True)) / \
        #                 (data.max(axis=(0, 1), keepdims=True) - data.min(axis=(0, 1), keepdims=True))

        # mean = np.mean(data, axis=(1), keepdims=True)
        # std = np.std(data, axis=(1), keepdims=True)
        # data_transformed = (data - mean) / (std + 1e-5)  # 防止分母为 0


        return data_transformed
    
    def fft(self,xi):
        data = xi
        patch,C = data.shape
        # data = data.reshape(-1,C)
        data, _ = computeFFT(
        data, n=self.args.num_t_pints)
        # data = data.reshape(clips,L//2,C)
        return data
    
    def torch_cwt(self, data, wavelet='cmor', scales=None):
        """
        对 PyTorch 张量进行连续小波变换 (CWT)
        :param data: 输入数据，形状为 N*C*T 的 Torch 张量
        :param wavelet: 小波类型
        :param scales: 小波尺度
        :return: 输出系数，形状为 N*C*len(scales)*T 的 Torch 张量
        """

        if scales is None:
            scales = np.arange(1, 50)  # 默认尺度范围
        else:
            scales = np.arange(1, scales+1)

        t,c = data.shape
        coeffs = np.zeros((c, len(scales), t))  # 初始化输出张量

        # 对每个样本和通道进行变换

        for j in range(c):
            # 将 Torch 张量转换为 Numpy 数组
            signal = data[:, j]
            # 执行 CWT
            cwt_coeff, _ = pywt.cwt(signal, scales, wavelet)
            # 将结果转回 Torch 张量
            coeffs[j, :, :] = cwt_coeff
        
        coeffs = coeffs.reshape((c* len(scales), t))

        return coeffs
    
    def use_cwt2(self, data, wavelet='cmor1.0-1.0', decimation = False, averaging = False, n_time_downsampled = 10, decim_filter = 'iir'):
        #print(f"data.shape: {data.shape}")
        _, n_channels = data.shape
        
        fs = 590
        sampling_period = 1/fs
        frequencies = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 85, 90, 95, 100, 110, 125, 135, 150, 175, 200]
        n_frequencies = len(frequencies)

        if decimation or averaging:
            transformed_data = np.zeros((n_channels, n_frequencies,  n_time_downsampled))
        else:
            transformed_data = np.zeros((n_channels, n_frequencies,  fs))

        # Define the wavelet and frequencies
        frequencies = np.array(frequencies)
        # Define frequency bins

        # Perform CWT on each channel and downsample
        for j in range(n_channels):
            # Perform CWT on the signal for this channel
            coef, _ = pywt.cwt(data[:, j], scales=1/(frequencies*sampling_period), wavelet=wavelet)
            if 'cmor' in wavelet or 'cgau' in wavelet:
                coef = np.abs(coef)
            #print(f"coef: {coef}") has real and imaginary parts
            #print(f"coef.shape: {coef.shape}") (24, 590)
            #if decimation is not None:
                # Decimate (downsample) the time dimension to the target size
                #transformed_data[j, :, :] = coef[:, ::int(fs/n_time_downsampled)][:, :n_time_downsampled] #(24,10)
                if averaging:
                    for i, freq_band in enumerate(coef):
                        for k in range(n_time_downsampled):
                            start = k * int(fs/n_time_downsampled)
                            end = start + int(fs/n_time_downsampled)
                            transformed_data[j, i, k] = np.mean(freq_band[start:end])
                elif decimation:
                    # Decimate the time dimension for each frequency band
                    for i, freq_band in enumerate(coef):
                        #print(f"decimate: {decimate(freq_band, int(fs / n_time_downsampled), ftype = decim_filter)}")
                        transformed_data[j, i, :] = decimate(freq_band, int(fs/n_time_downsampled), ftype = decim_filter)#inputting to np array only keeps real part
                        #print(f"transformed_data: {transformed_data[0]}")
                else:
                    transformed_data[j, :, :] = coef
                    #print(f"transformed_data: {transformed_data[0]}")
                #print(f"\ntransformed_data[j, :, :]: {transformed_data[j, :, :].shape}")


        #transformed_data = transformed_data.reshape((n_channels*n_frequencies, n_time_downsampled))
        #print(f"transformed_data.shape: {transformed_data.shape}")

        return transformed_data

    def band_pass(self, data):
        bounds = [ (4, 8), (8, 13), (13, 30), (30, 60), (60, 100), (100,200)]

        # bounds = [(1,4), (4, 13), (13, 60), (60, 200)]

        # bounds = [(30, 200)]


        # 滤波并堆叠结果
        filtered_data = multi_band_filter(data, bounds, 590)

        return filtered_data.transpose(0,2,1)

        

    def __getitem__(self, idx):
        # t0 = time.time()
        current_xi = self.x[idx]

        
        if self.args.dataset == "HAR":
            channels,length = current_xi.shape
        # print(current_xi.shape)
        # print(current_xi.shape)
        
        if self.args.model == "LSTM":
            length,channels = current_xi.shape
            current_xi = current_xi.reshape(self.args.max_clip_length,-1,channels)
        elif self.args.dataset == "BSIsamplewavelet":
            C, F, T = current_xi.shape
            if self.args.model == "BSIformerT":
                current_xi = current_xi.reshape((T,F*C))
            else:
                current_xi = current_xi.reshape((C,F*T),order='F')
        elif self.args.use_wavelet:
            current_xi = self.torch_cwt(current_xi, scales = self.args.wavelet_scale)
        elif self.args.use_bandpass:
            current_xi = self.band_pass(current_xi)
        elif self.args.use_cwt: #debug
            current_xi = self.use_cwt2(current_xi, wavelet=self.args.wavelet, decimation= self.args.decimation , averaging = self.args.averaging
                                    ,n_time_downsampled=self.args.n_time_downsampled, decim_filter=self.args.decim_filter)
            #print(f"in Dataset.py with current_xi.shape: {current_xi.shape}") #debug (31 ,24, 590)
            C, F, T = current_xi.shape
            if self.args.model == "BSIformerT":
                current_xi = current_xi.reshape((T, F*C))
                #print(f"in Dataset.py with current_xi.shape after reshape: {current_xi.shape}") #debug (590, 744)
                if self.args.pca:
                    pca = PCA(n_components=self.args.pca_features)

            else:
                current_xi = current_xi.reshape((C,F*T),order='F')

        else:
            length,channels = current_xi.shape
            current_xi = current_xi.reshape(channels* self.args.num_cut,-1)

        if self.args.use_fft:
            current_xi = self.fft(current_xi)
        
        # if self.args.scaling:
        #     current_xi = self.scaling(current_xi)

        if self.args.augmentation and self.split == "train":
            current_xi = self.adding_noise(current_xi)

        # current_xi = torch.Tensor(current_xi)

        y = self.y[idx]
        # print("task: ",self.args.task)
        if self.args.task == "Classification" or self.args.task == "SSLJoint" or self.args.task == "SSLEval":
            y = torch.LongTensor([y])
        else:
            y = torch.FloatTensor([y])

        return current_xi, y