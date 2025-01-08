import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import os

name = "EC03"

folder_path = 'src/data/'+name

save_path = 'src/data/pics_of_'+name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

train = np.load(folder_path+'/SSL/train_x.npy')
valid = np.load(folder_path+'/SSL/vali_x.npy')
test = np.load(folder_path+'/SSL/test_x.npy')



if name == "HAR":
    train = train.reshape((train.shape[0],-1,train.shape[1]))
    valid = valid.reshape((valid.shape[0],-1,valid.shape[1]))
    test = test.reshape((test.shape[0],-1,test.shape[1]))

train = train[:,0,:].reshape((train.shape[0],-1))
valid = valid[:,0,:].reshape((valid.shape[0],-1))
test = test[:,0,:].reshape((test.shape[0],-1))

for i in range(train[:,:30].shape[1]):

    train_i = train[:,i]
    valid_i = valid[:,i]
    test_i = test[:,i]


    print(train_i.min(),train_i.max(),train_i.mean())

    label = ["train", "valid", "test"]

    plt.figure()

    sns.violinplot(data = [train_i, valid_i, test_i])

    plt.xticks(ticks = [0, 1, 2], labels = label, fontsize = 11)

    print("saving pic to: ",save_path+str(i)+".jpg")

    plt.savefig(save_path+str(i)+".jpg")

    plt.close()

