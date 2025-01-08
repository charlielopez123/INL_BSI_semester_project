from aeon.datasets import load_classification
import numpy as np
import os
from tqdm import tqdm
import xarray as xr
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(pats_ids_in, lp, n_chans_all=64, test_day=None, tlim=[-1,1], event_types=['rest','move']):
    '''
    Load ECoG data from all subjects and combine (uses xarray variables)
    
    If len(pats_ids_in)>1, the number of electrodes will be padded or cut to match n_chans_all
    If test_day is not None, a variable with test data will be generated for the day specified
        If test_day = 'last', the last day will be set as the test day.
    '''
    if not isinstance(pats_ids_in, list):
        pats_ids_in = [pats_ids_in]
    sbj_order,sbj_order_test = [],[]
    X_test_subj,y_test_subj = [],[] #placeholder vals
        
    #Gather each subjects data, and concatenate all days
    for j in tqdm(range(len(pats_ids_in))):
        pat_curr = pats_ids_in[j]
        ep_data_in = xr.open_dataset(lp+pat_curr+'_ecog_data.nc')
        ep_times = np.asarray(ep_data_in.time)
        time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]
        n_ecog_chans = (len(ep_data_in.channels)-1)
        
        if test_day == 'last':
            test_day_curr = np.unique(ep_data_in.events)[-1] #select last day
        else:
            test_day_curr = test_day
        
        if n_chans_all < n_ecog_chans:
            n_chans_curr = n_chans_all
        else:
            n_chans_curr = n_ecog_chans
            
        
        
        days_all_in = np.asarray(ep_data_in.events)
        
        if test_day is None:
            #No test output here
            days_train = np.unique(days_all_in)
            test_day_curr = None
        else:
            days_train = np.unique(days_all_in)[:-1]
            day_test_curr = np.unique(days_all_in)[-1]
            days_test_inds = np.nonzero(days_all_in==day_test_curr)[0]
            
        #Compute indices of days_train in xarray dataset
        days_train_inds = []
        for day_tmp in list(days_train):
            days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])
        
        #Extract data and labels
        dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_curr),
                                    time=time_inds)].to_array().values.squeeze()
        labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                       time=0)].to_array().values.squeeze()
        sbj_order += [j]*dat_train.shape[0]
        
        if test_day is not None:
            dat_test = ep_data_in[dict(events=days_test_inds,channels=slice(0,n_chans_curr),
                                       time=time_inds)].to_array().values.squeeze()
            labels_test = ep_data_in[dict(events=days_test_inds,channels=ep_data_in.channels[-1],
                                          time=0)].to_array().values.squeeze()
            sbj_order_test += [j]*dat_test.shape[0]
            
        #Pad data in electrode dimension if necessary
        if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
            dat_sh = list(dat_train.shape)
            dat_sh[1] = n_chans_all
            #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
            X_pad = np.zeros(dat_sh)
            X_pad[:,:n_ecog_chans,...] = dat_train
            dat_train = X_pad.copy()
            
            if test_day is not None:
                dat_sh = list(dat_test.shape)
                dat_sh[1] = n_chans_all
                #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                X_pad = np.zeros(dat_sh)
                X_pad[:,:n_ecog_chans,...] = dat_test
                dat_test = X_pad.copy()
        
        #Concatenate across subjects
        if j==0:
            X_subj = dat_train.copy()
            y_subj = labels_train.copy()
            if test_day is not None:
                X_test_subj = dat_test.copy()
                y_test_subj = labels_test.copy()
        else:
            X_subj = np.concatenate((X_subj,dat_train.copy()),axis=0)
            y_subj = np.concatenate((y_subj,labels_train.copy()),axis=0)
            if test_day is not None:
                X_test_subj = np.concatenate((X_test_subj,dat_test.copy()),axis=0)
                y_test_subj = np.concatenate((y_test_subj,labels_test.copy()),axis=0)
    
    sbj_order = np.asarray(sbj_order)
    sbj_order_test = np.asarray(sbj_order_test)
    print('Data loaded!')
    
    return X_subj,y_subj,X_test_subj,y_test_subj,sbj_order,sbj_order_test


def splitting_data(name, X_train,Y_train, X_test,Y_test):

    
    le = LabelEncoder()
    train_sample = X_train
    test_sample = X_test
    
    train_labels=le.fit_transform(Y_train)
    test_labels = le.fit_transform(Y_test)

    
    

    # ssl_train_y_label = train_y_l[1:]
    
    # ssl_labels = []

    # for i in range(len(ssl_train_x_label)):
    #     if ssl_train_x_label[i] == ssl_train_y_label[i]:
    #         ssl_labels.append(i)
    # print("before cleaning: ",len(ssl_train_x_label)," after cleaning: ",len(ssl_labels))
    # print(ssl_train_x_label)

    

    # ssl_train = int(0.7*ssl_train_x.shape[0])
    # ssl_vali = int(1*ssl_train_x.shape[0])

    # train_ssl_x, vali_ssl_x, test_ssl_x = ssl_train_x[:ssl_train], ssl_train_x[ssl_train:ssl_vali], test_x_l[:-1]
    # train_ssl_y, vali_ssl_y, test_ssl_y = ssl_train_y[:ssl_train], ssl_train_y[ssl_train:ssl_vali], test_x_l[1:]
    # train_ssl_x_labels, vali_ssl_x_labels, test_ssl_x_labels = ssl_train_x_label[:ssl_train], ssl_train_x_label[ssl_train:ssl_vali], test_y_l[:-1]
    # print("ssl train shape: ",train_ssl_x.shape,train_ssl_x_labels.shape)
    # print("ssl vali shape: ",vali_ssl_x.shape,vali_ssl_x_labels.shape)
    # print("ssl test shape: ",test_ssl_x.shape,test_ssl_x_labels.shape)
    
    
    indies = np.arange(len(train_sample))
    np.random.shuffle(indies)

    train_x = train_sample[indies]
    train_y = train_labels[indies]

    print(train_y[:40])

    print("train_vali shape: ",train_x.shape,train_y.shape)


    train = int(0.8*train_x.shape[0])
    vali = int(1*train_x.shape[0])

    train_sample,vali_sample = train_x[:train],train_x[train:vali]
    train_labels,vali_labels = train_y[:train],train_y[train:vali]

    print("train shape: ",train_sample.shape,train_labels.shape)
    print("vali shape: ",vali_sample.shape,vali_labels.shape)

    print("test shape: ",test_sample.shape,test_labels.shape)

    folder_path = 'src/data/'+name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(folder_path+'/SSL')
        os.makedirs(folder_path+'/Detection')
    
    np.save(folder_path+'/SSL/train_x.npy',train_sample)
    np.save(folder_path+'/SSL/train_y.npy',train_sample)
    np.save(folder_path+'/SSL/train_x_labels.npy',train_labels)

    np.save(folder_path+'/SSL/vali_x.npy',vali_sample)
    np.save(folder_path+'/SSL/vali_y.npy',vali_sample)
    np.save(folder_path+'/SSL/vali_x_labels.npy',vali_labels)

    np.save(folder_path+'/SSL/test_x.npy',test_sample)
    np.save(folder_path+'/SSL/test_y.npy',test_sample)
    np.save(folder_path+'/SSL/test_x_labels.npy',test_labels)

    np.save(folder_path+'/Detection/train_x.npy',train_sample)
    np.save(folder_path+'/Detection/train_x_labels.npy',train_labels)

    np.save(folder_path+'/Detection/vali_x.npy',vali_sample)
    np.save(folder_path+'/Detection/vali_x_labels.npy',vali_labels)

    np.save(folder_path+'/Detection/test_x.npy',test_sample)
    np.save(folder_path+'/Detection/test_x_labels.npy',test_labels)


X_subj,y_subj,X_test_subj,y_test_subj,_,_ = load_data('EC03','src/data/',test_day='last')

print(X_subj.shape, X_test_subj.shape)

splitting_data('EC03',X_subj.transpose((0,2,1)),y_subj,X_test_subj.transpose((0,2,1)),y_test_subj)

