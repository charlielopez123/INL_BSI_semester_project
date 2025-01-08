from aeon.datasets import load_classification
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(name):

    X, label, _ = load_classification(name, return_metadata=True)
    le = LabelEncoder()
    label=le.fit_transform(label)

    train_x_l, _ = load_classification(name, split="train")
    
    scaler = StandardScaler()
    scaler.fit(train_x_l.reshape(-1,1))

    l,c,d = train_x_l.shape
    
    train_x_l = train_x_l.reshape(-1,1)
    train_x_l = scaler.transform(train_x_l)

    max_val = np.abs(train_x_l).max()
    # train_x_l = train_x_l/max_val
    train_x_l = train_x_l.reshape((l,d,c))

    train_y_l = label[:l]

    test_x_l, _ = load_classification(name, split="test")
    l,c,d = test_x_l.shape
    
    test_x_l = test_x_l.reshape(-1,1)
    test_x_l = scaler.transform(test_x_l)
    # test_x_l = test_x_l/max_val
    test_x_l = test_x_l.reshape((l,d,c))

    test_y_l = label[-l:]

    
    
    ssl_train_x = train_x_l[:-1]
    ssl_train_y = train_x_l[1:]
    ssl_train_x_label = train_y_l[:-1]

    # ssl_train_y_label = train_y_l[1:]
    
    # ssl_labels = []

    # for i in range(len(ssl_train_x_label)):
    #     if ssl_train_x_label[i] == ssl_train_y_label[i]:
    #         ssl_labels.append(i)
    # print("before cleaning: ",len(ssl_train_x_label)," after cleaning: ",len(ssl_labels))
    # print(ssl_train_x_label)

    
    indies = np.arange(len(ssl_train_x))
    np.random.shuffle(indies)
    ssl_train_x = ssl_train_x[indies]
    ssl_train_y = ssl_train_y[indies]
    ssl_train_x_label = ssl_train_x_label[indies]

    

    ssl_train = int(0.7*ssl_train_x.shape[0])
    ssl_vali = int(1*ssl_train_x.shape[0])

    train_ssl_x, vali_ssl_x, test_ssl_x = ssl_train_x[:ssl_train], ssl_train_x[ssl_train:ssl_vali], test_x_l[:-1]
    train_ssl_y, vali_ssl_y, test_ssl_y = ssl_train_y[:ssl_train], ssl_train_y[ssl_train:ssl_vali], test_x_l[1:]
    train_ssl_x_labels, vali_ssl_x_labels, test_ssl_x_labels = ssl_train_x_label[:ssl_train], ssl_train_x_label[ssl_train:ssl_vali], test_y_l[:-1]
    print("ssl train shape: ",train_ssl_x.shape,train_ssl_x_labels.shape)
    print("ssl vali shape: ",vali_ssl_x.shape,vali_ssl_x_labels.shape)
    print("ssl test shape: ",test_ssl_x.shape,test_ssl_x_labels.shape)
    
    
    indies = np.arange(len(train_x_l))
    np.random.shuffle(indies)

    train_x = train_x_l[indies]
    train_y = train_y_l[indies]

    print("train_vali shape: ",train_x.shape,train_y.shape)


    train = int(0.85*train_x.shape[0])
    vali = int(1*train_x.shape[0])

    train_sample,vali_sample = train_x[:train],train_x[train:vali]
    train_labels,vali_labels = train_y[:train],train_y[train:vali]

    print("train shape: ",train_sample.shape,train_labels.shape)
    print("vali shape: ",vali_sample.shape,vali_labels.shape)


    test_sample = test_x_l
    test_labels = test_y_l

    print("test shape: ",test_sample.shape,test_labels.shape)

    folder_path = 'src/data/'+name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(folder_path+'/SSL')
        os.makedirs(folder_path+'/Classification')
    
    print(train_ssl_x.min(),train_ssl_x.max(),train_ssl_x.mean())

    print(vali_ssl_x.min(),vali_ssl_x.max(),vali_ssl_x.mean())



    np.save(folder_path+'/SSL/train_x.npy',train_ssl_x)
    np.save(folder_path+'/SSL/train_y.npy',train_ssl_y)
    np.save(folder_path+'/SSL/train_x_labels.npy',train_ssl_x_labels)

    np.save(folder_path+'/SSL/vali_x.npy',vali_ssl_x)
    np.save(folder_path+'/SSL/vali_y.npy',vali_ssl_y)
    np.save(folder_path+'/SSL/vali_x_labels.npy',vali_ssl_x_labels)

    np.save(folder_path+'/SSL/test_x.npy',test_ssl_x)
    np.save(folder_path+'/SSL/test_y.npy',test_ssl_y)
    np.save(folder_path+'/SSL/test_x_labels.npy',test_ssl_x_labels)

    np.save(folder_path+'/Classification/train_x.npy',train_sample)
    np.save(folder_path+'/Classification/train_x_labels.npy',train_labels)

    np.save(folder_path+'/Classification/vali_x.npy',vali_sample)
    np.save(folder_path+'/Classification/vali_x_labels.npy',vali_labels)

    np.save(folder_path+'/Classification/test_x.npy',test_sample)
    np.save(folder_path+'/Classification/test_x_labels.npy',test_labels)




load_data('MotorImagery')