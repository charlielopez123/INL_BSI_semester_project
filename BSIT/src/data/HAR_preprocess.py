import pandas as pd
import numpy as np
import torch
import random 
# import constants
from typing import Any, Callable, List, Optional, Tuple, Union
from numpy.typing import NDArray
from collections import Counter
import tqdm
import os
from seiz_eeg.dataset import EEGDataset, to_arrays
import matplotlib.pyplot as plt

har_train = torch.load("src/data/HAR/train.pt")
har_vali = torch.load("src/data/HAR/val.pt")
har_test = torch.load("src/data/HAR/test.pt")

train_sample,train_labels = har_train["samples"], har_train["labels"]
vali_sample,vali_labels = har_vali["samples"], har_vali["labels"]
test_sample,test_labels = har_test["samples"], har_test["labels"]

np.savetxt('src/data/HAR/train_labels.txt',train_labels)
np.savetxt('src/data/HAR/vali_labels.txt',vali_labels)
np.savetxt('src/data/HAR/test_labels.txt',test_labels)


detection_train_labels = (train_labels.numpy()>2).astype(int)
detection_vali_labels = (vali_labels.numpy()>2).astype(int)
detection_test_labels = (test_labels.numpy()>2).astype(int)


# train_labels = train_labels - 1
# vali_labels = vali_labels - 1
# test_labels = test_labels - 1


train_length = train_sample.shape[0]


label_counts = Counter(train_labels.numpy())

# Print the counts
for label, count in label_counts.items():
    print(f'Label {label}: {count} occurrences')
print(" ")


label_counts = Counter(detection_train_labels)

# Print the counts
for label, count in label_counts.items():
    print(f'Label {label}: {count} occurrences')
print(" ")

# plot_sample = sample[:1000,0,:]
# plot_sample = plot_sample.reshape(-1)
# print(plot_sample.shape)

subject_train_vali = np.loadtxt('src/data/HAR/subject_train.txt')
subject_test = np.loadtxt('src/data/HAR/subject_test.txt')

print("subject info",subject_test.shape)
label_counts = Counter(subject_test)

# Print the counts
for label, count in label_counts.items():
    print(f'subject {label}: {count} occurrences')

def split_SSL_group(data,subjects):
    subject_data = []
    current_subject = -1
    # print("datashape: ",data.shape,data[:100].shape)
    i_start = 0
    for i,subject in enumerate(subjects):
        if current_subject == -1:
            current_subject = subject
        if current_subject != subject:
            subject_data.append(data[i_start:i-1])
            print("store subject: ",current_subject,"from: ",i_start," to: ",i-1," num: ",i-1-i_start+1)
            i_start = i
            current_subject = subject
    subject_data.append(data[i_start:i])
    print("store subject: ",current_subject,"from: ",i_start," to: ",i," num: ",i-i_start+1)
    print(" ")
    return subject_data

def generate_SSL_pairs(data_group):
    ssl_x_list = []
    ssl_y_list = []
    for member in data_group:
        ssl_x_list.append(member[:-1])
        ssl_y_list.append(member[1:])
    
    ssl_x = np.concatenate(ssl_x_list,axis=0)
    ssl_y = np.concatenate(ssl_y_list,axis=0)
    print("ssl shape: ",ssl_x.shape,ssl_y.shape)
    print(" ")
    return ssl_x,ssl_y
    


train_vali_sample,train_vali_labels = torch.cat([train_sample,vali_sample]),torch.cat([train_labels,vali_labels])
print("train vali: ",train_vali_sample.shape,train_vali_labels.shape)
        
# test_group = split_SSL_group(test_sample,subject_test)
train_vali_group = split_SSL_group(train_vali_sample,train_vali_labels)
train_vali_label_group = split_SSL_group(train_vali_labels,train_vali_labels)
test_group = split_SSL_group(test_sample,test_labels)
test_label_group = split_SSL_group(test_labels,test_labels)

train_vali_ssl_x ,train_vali_ssl_y= generate_SSL_pairs(train_vali_group)
train_vali_ssl_x_labels ,_= generate_SSL_pairs(train_vali_label_group)
test_ssl_x,test_ssl_y = generate_SSL_pairs(test_group)
test_ssl_x_labels,_ = generate_SSL_pairs(test_label_group)

train_ssl_x,vali_ssl_x = train_vali_ssl_x[:train_length],train_vali_ssl_x[train_length:]
train_ssl_y,vali_ssl_y = train_vali_ssl_y[:train_length],train_vali_ssl_y[train_length:]
train_ssl_x_labels,vali_ssl_x_labels = train_vali_ssl_x_labels[:train_length],train_vali_ssl_x_labels[train_length:]

np.save('src/data/HAR/SSL/train_x.npy',train_ssl_x)
np.save('src/data/HAR/SSL/train_y.npy',train_ssl_y)
np.save('src/data/HAR/SSL/train_x_labels.npy',train_ssl_x_labels)

np.save('src/data/HAR/SSL/vali_x.npy',vali_ssl_x)
np.save('src/data/HAR/SSL/vali_y.npy',vali_ssl_y)
np.save('src/data/HAR/SSL/vali_x_labels.npy',vali_ssl_x_labels)

np.save('src/data/HAR/SSL/test_x.npy',test_ssl_x)
np.save('src/data/HAR/SSL/test_y.npy',test_ssl_y)
np.save('src/data/HAR/SSL/test_x_labels.npy',test_ssl_x_labels)

np.save('src/data/HAR/Classification/train_x.npy',train_sample)
np.save('src/data/HAR/Classification/train_x_labels.npy',train_labels)

np.save('src/data/HAR/Classification/vali_x.npy',vali_sample)
np.save('src/data/HAR/Classification/vali_x_labels.npy',vali_labels)

np.save('src/data/HAR/Classification/test_x.npy',test_sample)
np.save('src/data/HAR/Classification/test_x_labels.npy',test_labels)

np.save('src/data/HAR/Detection/train_x.npy',train_sample)
np.save('src/data/HAR/Detection/train_x_labels.npy',detection_train_labels)

np.save('src/data/HAR/Detection/vali_x.npy',vali_sample)
np.save('src/data/HAR/Detection/vali_x_labels.npy',detection_vali_labels)

np.save('src/data/HAR/Detection/test_x.npy',test_sample)
np.save('src/data/HAR/Detection/test_x_labels.npy',detection_test_labels)



# print()



# plt.plot(range(plot_sample.shape[0]),plot_sample)

# plt.savefig("HAR_plot.jpg")