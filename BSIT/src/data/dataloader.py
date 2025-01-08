import numpy as np
import pandas as pd
import torch
from data.data_utils import signal_transform, binary_label_transform, batch_bandpass_filter, MNEFilter
# from seiz_eeg.dataset import EEGDataset,EEGFileDataset,to_arrays
from data.Dataset import BSI_Dataset, Fingerflex_Dataset
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
import time
import os
from torch_geometric.datasets import TUDataset
import constants
import psutil
import scipy.signal as signal
import random
from sklearn.decomposition import PCA




def create_regression_loader(args,split,shuffle, task_op = None):
    seed = args.seed
    torch.manual_seed(seed)
    if task_op is not None:
        TASK = task_op
    else:
        TASK = args.task
    dataset = args.dataset


    if split == "train":
        x = np.load('src/data/'+dataset+'/Regression/train_x.npy').astype(np.float32)
        y = np.load('src/data/'+dataset+'/Regression/train_y.npy').astype(np.float32)
    elif split == "valid":
        x = np.load('src/data/'+dataset+'/Regression/valid_x.npy').astype(np.float32)
        y = np.load('src/data/'+dataset+'/Regression/valid_y.npy').astype(np.float32)
    else:
        x = np.load('src/data/'+dataset+'/Regression/test_x.npy').astype(np.float32)
        y = np.load('src/data/'+dataset+'/Regression/test_y.npy').astype(np.float32)
    
    print("create ",split," set: ",x.shape[0])

    dataset = Fingerflex_Dataset(x,y,args,split)
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)

def create_non_graph_loader(args,split,shuffle, task_op = None):
    seed = args.seed
    torch.manual_seed(seed)
    if task_op is not None:
        TASK = task_op
    else:
        TASK = args.task
    dataset = args.dataset

    if TASK == "SSLEval":
        if dataset == "HAR":
            TASK = "Classification"
        else:
            TASK = "Detection"
    
    if TASK == "Classification":
        if split == "train":
            x = np.load('src/data/'+dataset+'/Classification/train_x.npy')
            y = np.load('src/data/'+dataset+'/Classification/train_x_labels.npy')
        elif split == "valid":
            x = np.load('src/data/'+dataset+'/Classification/vali_x.npy')
            y = np.load('src/data/'+dataset+'/Classification/vali_x_labels.npy')
        else:
            x = np.load('src/data/'+dataset+'/Classification/test_x.npy')
            y = np.load('src/data/'+dataset+'/Classification/test_x_labels.npy')

    else:
        if split == "train":
            x = np.load('src/data/'+dataset+'/Detection/train_x.npy')
            y = np.load('src/data/'+dataset+'/Detection/train_x_labels.npy')
        elif split == "valid":
            x = np.load('src/data/'+dataset+'/Detection/vali_x.npy')
            y = np.load('src/data/'+dataset+'/Detection/vali_x_labels.npy')
        else:
            x = np.load('src/data/'+dataset+'/Detection/test_x.npy')
            y = np.load('src/data/'+dataset+'/Detection/test_x_labels.npy')

    
    print("create ",split," set: ",x.shape[0])

    dataset = BSI_Dataset(x,y,args,split)
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)

def create_BSI_loader(args,split,scaler, shuffle, task_op = None):
    seed = args.seed
    torch.manual_seed(seed)
    
    if args.dataset == "BSI250":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_f250.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_f250.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_f250.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_f250.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_f250.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_f250.npy')

    elif args.dataset == "BSI":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed.npy')
    
    elif args.dataset == "BSI1s":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_1s.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_1s.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_1s.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_1s.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_1s.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_1s.npy')

    elif args.dataset == "BSI2s":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_2s.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_2s.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_2s.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_2s.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_2s.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_2s.npy')


    elif args.dataset == "BSI5class":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_processed_5class.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_processed_5class.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_5class.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_5class.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_processed_5class.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_processed_5class.npy')
    
    elif args.dataset == "BSIaligned":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_aligned1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_aligned1_2.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_aligned1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_aligned1_2.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_aligned1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_aligned1_2.npy')


    elif args.dataset == "BSIunaligned":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_unaligneds1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_unaligneds1_2.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_unaligneds1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_unaligneds1_2.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_unaligneds1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_unaligneds1_2.npy')
    
    elif args.dataset == "BSIalignedus":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_alignedus1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_alignedus1_2.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_alignedus1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_alignedus1_2.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_alignedus1_2.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_alignedus1_2.npy')

    elif args.dataset == "BSIstreamaligned":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train_streamaligned.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train_streamaligned.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_streamaligned.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_streamaligned.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test_streamaligned.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test_streamaligned.npy')

    elif args.dataset == "BSI+onset":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train+onset.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train+onset.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test+onset.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test+onset.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test+onset.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test+onset.npy')

    elif args.dataset == "BSI+onset+una":
        if split == "train":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_train+onset+una.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_train+onset+una.npy')
        elif split == "valid":
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test+onset+una.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test+onset+una.npy')
        else:
            x = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/X_test+onset+una.npy')
            y = np.load('/run/user/276352/gvfs/smb-share:server=upcourtinenas,share=upperlimb/INL/dataset_1025/label_test+onset+una.npy')

    elif args.dataset == "BSIsample":
        if split == "train":
            x = np.load(args.dataset_dir + 'X_train_samplew1s01.npy')
            y = np.load(args.dataset_dir + 'label_train_samplew1s01.npy')
        elif split == "valid":
            x = np.load(args.dataset_dir + 'X_test_samplew1s01.npy')
            y = np.load(args.dataset_dir + 'abel_test_samplew1s01.npy')
        else:
            x = np.load(args.dataset_dir + 'X_test_samplew1s01.npy')
            y = np.load(args.dataset_dir + 'label_test_samplew1s01.npy')

    elif args.dataset == "BSIsamplewavelet":
        if split == "train":
            x = np.load(args.dataset_dir + 'X_train_samplewavelet.npy')
            y = np.load(args.dataset_dir + 'label_train_samplewavelet.npy')
        elif split == "valid":
            x = np.load(args.dataset_dir + 'X_test_samplewavelet.npy')
            y = np.load(args.dataset_dir + 'label_test_samplewavelet.npy')
        else:
            x = np.load(args.dataset_dir + 'X_test_samplewavelet.npy')
            y = np.load(args.dataset_dir + 'label_test_samplewavelet.npy')

    if args.dataset != "BSIsamplewavelet":   
        x = x.transpose((0,2,1))
    
    # non_zero_indices = [i for i, value in enumerate(y) if value != 0]
    # y = y[non_zero_indices]
    # x = x[non_zero_indices]
    # y = y-1

    
    unique_values, counts = np.unique(y, return_counts=True)

    # 打印结果
    print("labels: ", unique_values)
    print("nums: ", counts)

    

    if args.preprocess:

        filter = MNEFilter(sfreq=590, l_freq=1, h_freq=200, notch_freqs=np.arange(50, 201, 50), apply_car=True)

        N,T,D = x.shape
        x = x.reshape((T,-1))
        x = filter.forward(x)
        x = x.reshape((N,T,D))
    
    if args.scaling and not args.use_fft:
        xshape = x.shape
        data_channel = x.reshape(x.shape[0],-1)

        whole_data = scaler.transform(data_channel)
        x = whole_data.reshape(xshape)
    
    if args.use_fft and args.scaling:

        fft_result = np.fft.fft(x, axis=-1)
        abs_result = np.abs(fft_result[:, :, :x.shape[1] // 2])
        xfft = np.log(abs_result + 1e-8)  # 防止对零取对数

        xshape = xfft.shape
        data_channel = xfft.reshape(xfft.shape[0],-1)

        whole_data = scaler.transform(data_channel)
        x = whole_data.reshape(xshape)
        

    
    # x = x[:,::2,:]
        
    # x = signal.resample(x,args.num_t_pints,axis=1)
    # print("down sampled: ",x.shape)

    # x = batch_bandpass_filter(x, 1, 200, 590)

    # if split == "train":

    #     indices_y0 = np.where(y == 0)[0]
    #     indices_y_other = np.where(y != 0)[0]

    #     # 计算需要保留的 y=0 样本数量（20%）
    #     n_samples_y0 = int(len(indices_y0) * 0.2)

    #     # 随机选择 y=0 的子集
    #     selected_indices_y0 = np.random.choice(indices_y0, n_samples_y0, replace=False)

    #     # 合并降采样后的 y=0 索引和其他标签的索引
    #     final_indices = np.concatenate([selected_indices_y0, indices_y_other])

    #     # 对索引进行打乱
    #     np.random.shuffle(final_indices)

    #     # 根据打乱后的索引获取降采样并打乱后的 x 和 y
    #     x = x[final_indices]
    #     y = y[final_indices]

    # indices_y0 = np.where(y == 0)[0]
    # indices_y_other = np.where(y != 0)[0]

    # x_rest = x[indices_y0]
    # x_rest = x_rest[:,::2,:]

    # x_move = x[indices_y_other]
    # x_move1 = x_move[:,::2,:]
    # x_move2 = x_move[:,1::2,:]

    # new_x = np.concatenate([x_rest, x_move1, x_move2])
    # new_y = np.concatenate([y[indices_y0],y[indices_y_other],y[indices_y_other]])

    #     # 生成索引
    # indices = np.arange(len(new_x))

    # # 打乱索引
    # np.random.shuffle(indices)

    # # 根据打乱后的索引获取降采样并打乱后的 x 和 y
    # x = new_x[indices]
    # y = new_y[indices]


    
    print("create ", split," set: ", x.shape)

    dataset = BSI_Dataset(x,y,args,split, scaler)
    #print(type(dataset)) #debug dataset length (5031)
    print(f"before dataset.shape: {dataset[0][0].shape}") #debug (5031)

    if args.pca:
        #Collect each X from train as to apply PCA
        # Step 1: Extract all `x` values
        # Assuming dataset[i][0] is shaped (T, F*C)
        all_x = []  # Collect all `x` values
        for i in range(len(dataset)):
            xi, _ = dataset[i]  # Unpack the tuple (x, y)
            all_x.append(xi)  # Shape of xi: (T, F*C)
        print("created all_x")

        # Concatenate along the time dimension (stack all time samples)
        all_x = np.vstack(all_x)  # Shape: (Total_T, F*C)
        print("finished_concatenation")

        # Step 2: Recreate the dataset with PCA-transformed x
        if split is "train":
            args.pca_transform = PCA(n_components=args.pca_features)
        all_x_pca = args.pca_transform.fit_transform(all_x)
        print("finished PCA")

        # Step 3: Split the PCA-transformed `x` back into individual samples
        start_idx = 0
        pca_transformed_data = []  # To store the (x, y) tuples
        for i in range(len(dataset)):
            xi, yi = dataset[i]
            T = xi.shape[0]  # Time dimension of the current sample
            x_pca = all_x_pca[start_idx:start_idx + T]  # Extract the transformed slice
            pca_transformed_data.append((x_pca, yi))  # Append the transformed (x, y) tuple
            start_idx += T

        del all_x
        del all_x_pca

        #print(type(dataset)) #debug 
        print(f"after pca_transformed_data[0][0].shape: {pca_transformed_data[0][0].shape}") #debug 
    
    if args.forward_selection:
        print("dataloader", split, "forward selection")
        selected_features_dataset = []
        for i in range(len(dataset)):
            xi, yi = dataset[i]
            selected_features_xi = xi[:, args.current_features]
            selected_features_dataset.append((selected_features_xi, yi))


    if args.feature_selection_wrapper:
        X = []
        y = []
        print("dataloader", split, "feature selection wrapper")
        selected_features_dataset = []
        #a = args.num_patches * [dataset[13][1]]
        #print(len(a))
        for i in range(0, len(dataset), 300):
            xi, yi = dataset[i]
            X.append(xi)
            y.append(args.num_patches * [yi])

        X = np.concatenate(X, axis=0)  # shape: (N, num_features)
        #print("X_shape", X.shape)
        y = np.concatenate(y, axis=0)    # shape: (N,)
        #print("y_shape", y.shape)

        return X, y
    
    
    
    g = torch.Generator()
    g.manual_seed(seed)

    
    if args.pca:
        return DataLoader(pca_transformed_data, batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)
    elif args.forward_selection:
        return DataLoader(selected_features_dataset, batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)
    else:
        return DataLoader(dataset,batch_size = args.batch_size,shuffle=shuffle,num_workers=args.N_WORKERS,prefetch_factor=args.prefetch_factor,generator = g)

