import numpy as np

from sklearn.decomposition import PCA
import os,sys

sys.path.append('/home/yuhxie/Digital-Bridge-Yuhan/src')


from visualization.embedding_viz import simple_PCA_vis

def vis_origin_data(name, split):
    dataset = name
    if split == "train":
        x = np.load('src/data/'+dataset+'/Detection/train_x.npy')
        y = np.load('src/data/'+dataset+'/Detection/train_x_labels.npy')
    elif split == "valid":
        x = np.load('src/data/'+dataset+'/Detection/vali_x.npy')
        y = np.load('src/data/'+dataset+'/Detection/vali_x_labels.npy')
    else:
        x = np.load('src/data/'+dataset+'/Detection/test_x.npy')
        y = np.load('src/data/'+dataset+'/Detection/test_x_labels.npy')
    
    
    x = (x -  np.mean(x, axis= -1,keepdims = True) )/np.std(x, axis= -1,keepdims = True)
    x = x.reshape((x.shape[0],-1)) 
    # print(x.shape)s

    pca = PCA(n_components=2).fit(x)
    simple_PCA_vis(pca,x,y, 'src/data/'+dataset+'/Detection/' ,split + '_PCA')

vis_origin_data('EC03','train')
    