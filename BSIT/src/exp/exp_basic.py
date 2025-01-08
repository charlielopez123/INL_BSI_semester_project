import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import utils
import pickle
from visualization.graph_viz import draw_graph_weighted_edge,get_spectral_graph_positions,draw_graph_chord_edge, draw_heat_graph, draw_disheat_graph
from visualization.embedding_viz import tSNE_vis,PCA_vis,LogME_score,EEG_PCA_vis,mux_PCA_vis
import random
from sklearn.decomposition import PCA
from metrics.SFDA import SFDA_Score
from metrics.NLEEP import NLEEP
import subprocess


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        if not(args.forward_selection) and not(args.feature_selection_wrapper):
            self.model = self._build_model().to(self.device)
        self.adj_mat_dir = args.adj_mat_dir
        self.pca = None
        self.pca2 = None
        self.mu_x_pca = None
        #self.branch_name = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            torch.cuda.current_device()

            torch.cuda._initialized = True
            # print(os.environ["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('{}'.format(self.args.gpu))
            print('Use GPU: {}'.format(self.args.gpu))
            print(os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def plot_graph(self,adj_mx,p_mx,w_mx,path):
        with open(self.adj_mat_dir, 'rb') as pf:
            adj_mat = pickle.load(pf)
        node_id_dict = {}
        for key, val in adj_mat[1].items():
            key = key.split(' ')[-1]
            node_id_dict[key] = val
        pos_spec = get_spectral_graph_positions(self.adj_mat_dir)
        draw_graph_weighted_edge(adj_mx, node_id_dict, pos_spec, is_directed=True, plot_colorbar=True, font_size=30, save_dir = path)
        draw_graph_chord_edge(adj_mx, node_id_dict, pos_spec, is_directed=True, plot_colorbar=True, font_size=30, save_dir = path)
        draw_heat_graph(p_mx,w_mx, node_id_dict, plot_colorbar=True, font_size=30, save_dir = path)
    
    def plot_mu_x_embedding(self,mu_x,wx,labels,epochs,path):

        if self.mu_x_pca is None:
            self.mu_x_pca = PCA(n_components=2).fit(wx)
        
        mux_PCA_vis(self.mu_x_pca,wx,labels,mu_x,epochs,path)


    def plot_embedding(self,embedding,labels,epochs,train_loss, path,ssl=None,embedding_dict = None, embedding_batch = None):

        if self.pca is None:
            self.pca = PCA(n_components=2).fit(embedding)

        if self.pca2 is None:
            indices = np.where((labels == 0) | (labels == 2) )[0]
            sub_embedding = embedding[indices]
            self.pca2 = PCA(n_components=2).fit(sub_embedding)
        
        self.plot_embedding_spectrum(embedding,epochs,path)

        
        if self.args.dataset == "Epilepsy":
            if self.args.task == "Detection":
                PCA_vis(self.pca,embedding,labels,epochs,train_loss, path,ssl)
            else:
                EEG_PCA_vis(self.pca,self.pca2, embedding,labels,epochs,train_loss, path,ssl, embedding_dict = embedding_dict, embedding_batch = embedding_batch)
        else:

            PCA_vis(self.pca,embedding,labels,epochs,train_loss, path,ssl,embedding_dict = embedding_dict, embedding_batch = embedding_batch)
        
        tSNE_vis(embedding,labels,epochs,train_loss, path,ssl,embedding_dict = embedding_dict, embedding_batch = embedding_batch)


        
    
    def plot_dist_graph(self):
        with open(self.adj_mat_dir, 'rb') as pf:
            adj_mat = pickle.load(pf)
        path = "./src/visualization"
        node_id_dict = {}
        adj_mx = adj_mat[2]
        print("dist: ",adj_mx.shape)
        for key, val in adj_mat[1].items():
            key = key.split(' ')[-1]
            node_id_dict[key] = val
        p_mx = np.array([1 if element != 0 else 0 for element in adj_mx.reshape(-1)])
        print("dist: ",p_mx.shape)
        p_mx = p_mx.reshape((19,19))
        print("dist: ",p_mx.shape)
        w_mx = adj_mx
        pos_spec = get_spectral_graph_positions(self.adj_mat_dir)
        draw_graph_weighted_edge(adj_mx, node_id_dict, pos_spec, is_directed=True, plot_colorbar=True, font_size=30, save_dir = path)
        draw_graph_chord_edge(adj_mx, node_id_dict, pos_spec, is_directed=True, plot_colorbar=True, font_size=30, save_dir = path)
        draw_disheat_graph(p_mx,w_mx, node_id_dict, plot_colorbar=True, font_size=30, save_dir = path)
            

    def plot_score(self, train, vali, test, path, args):
        
        fig = plt.figure(figsize = (7,5))
        # plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
        x = range(0,len(vali))
        learning_rate = args.learning_rate
        decay = args.weight_decay
        save_vali = np.array(vali)
        np.save(path+'npys/vali_score.npy',save_vali)
        save_test = np.array(test)
        np.save(path+'npys/test_score',save_test)
        print("ploting score: length: ",x)
        if len(train)>0:
            p1 = plt.plot(x, train,'r-', label = u'train')
            plt.legend()
            save_train = np.array(train)
            np.save(path+'npys/train_score.npy',save_train)
        p2 = plt.plot(x,test, 'b-', label = u'test')
        plt.legend()
        p3 = plt.plot(x,vali, 'g-', label = u'vali')
        plt.legend()
        plt.xlabel(u'epoches')
        plt.ylabel(u'value')
        if args.n_classes >2:
            title = 'F1 Score'
        else:
            title = 'AUROC Score'
        plt.title(title +', lr: '+str(learning_rate)+" decay: "+str(decay))
        plt.savefig(path+'score.jpg',dpi=300)
    
    def plot_loss(self, train, vali, test, path, args):
        
        fig = plt.figure(figsize = (7,5))
        # plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
        x = range(0,len(vali))
        learning_rate = args.learning_rate
        decay = args.weight_decay
        save_vali = np.array(vali)
        np.save(path+'npys/vali_loss.npy',save_vali)
        save_test = np.array(test)
        np.save(path+'npys/test_loss',save_test)
        print("ploting loss: length: ",x)
        if len(train)>0:
            save_train = np.array(train)
            np.save(path+'npys/train_loss.npy',save_train)
            p1 = plt.plot(x, train,'r-', label = u'train')
            plt.legend()
        p2 = plt.plot(x,test, 'b-', label = u'test')
        plt.legend()
        p3 = plt.plot(x,vali, 'g-', label = u'vali')
        plt.legend()
        plt.xlabel(u'epoches')
        plt.ylabel(u'loss')
        title = 'Training Loss'

        plt.title(title +', lr: '+str(learning_rate)+" decay: "+str(decay))
        plt.savefig(path+'loss.jpg',dpi=300)
    
    def plot_joint_loss(self, joint, sup, ssl, path, args):
        
        fig = plt.figure(figsize = (7,5))
        # plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
        x = range(0,len(joint))
        learning_rate = args.learning_rate
        decay = args.weight_decay
        print("ploting program: length: ",x)

        np.save(path+'npys/joint_loss.npy',np.array(joint))
        np.save(path+'npys/downstream_loss.npy',np.array(sup))
        np.save(path+'npys/auxiliary_loss.npy',np.array(ssl))
        
        p1 = plt.plot(x, joint,'r-', label = u'joint_loss')
        plt.legend()
        p2 = plt.plot(x,sup, 'b-', label = u'downstream_loss')
        plt.legend()
        p3 = plt.plot(x,ssl, 'g-', label = u'auxiliary_loss')
        plt.legend()
        plt.xlabel(u'epoches')
        plt.ylabel(u'loss')
        plt.title('Joint Learning Loss, lr: '+str(learning_rate)+" decay: "+str(decay))
        plt.savefig(path+'joint_loss.jpg')
    
    def plot_cluster_loss(self, cluster, diverge_dist, diverge_angle, path, args):
        
        fig = plt.figure(figsize = (7,5))
        # plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
        x = range(0,len(cluster))
        learning_rate = args.learning_rate
        decay = args.weight_decay
        print("ploting program: length: ",x)

        np.save(path+'npys/cluster_loss.npy',np.array(cluster))
        np.save(path+'npys/diverge_dist_loss.npy',np.array(diverge_dist))
        np.save(path+'npys/diverge_angle_loss.npy',np.array(diverge_angle))
        
        p1 = plt.plot(x, cluster,'r-', label = u'cluster')
        plt.legend()
        p2 = plt.plot(x, diverge_dist, 'b-', label = u'diverge dist')
        plt.legend()
        p3 = plt.plot(x, diverge_angle, 'g-', label = u'diverge angle')
        plt.legend()
        plt.xlabel(u'epoches')
        plt.ylabel(u'loss')
        plt.title('supervised clustering loss, lr: '+str(learning_rate)+" decay: "+str(decay))
        plt.savefig(path+'cluster_loss.jpg')
    
    def plot_SSL_metrics(self, LogME, SFDA, NLEEP, path, args):

        title_size = 15
        legend_size = 12
        label_size = 15
        
        fig = plt.figure(figsize = (7,5))
        # plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
        x = range(0,len(LogME))
        learning_rate = args.learning_rate
        decay = args.weight_decay
        print("ploting program: length: ",x)

        np.save(path+'npys/LogME.npy',np.array(LogME))
        np.save(path+'npys/SFDA.npy',np.array(SFDA))
        np.save(path+'npys/NLEEP.npy',np.array(NLEEP))
        
        p1 = plt.plot(x, LogME,'r-', label = u'LogME')
        p2 = plt.plot(x, SFDA,'b-', label = u'SFDA')
        p3 = plt.plot(x, NLEEP, 'g-', label= u'NLEEP')
        plt.legend(fontsize = legend_size)
        plt.xlabel(u'epochs',fontsize = label_size)
        plt.ylabel(u'SSL_metrics',fontsize = label_size)
        plt.ylim(-0.25,1.75)
        plt.title('SSL_metrics, lr: '+str(learning_rate)+" decay: "+str(decay),fontsize = title_size)
        plt.savefig(path+'SSL_metric.jpg',dpi=300)

    def plot_embedding_spectrum(self, embedding,epochs, save_dir):
        
        fig = plt.figure(figsize = (7,5))
        # plt.plot(x,y,'g-',label=u'Dense_Unet(block layer=5)')`
        if np.isnan(embedding).any():
            print("nan in embedding!",embedding)
        singular_value = self.singular(embedding)
        x = range(0,len(singular_value))
        print("ploting embedding spectrum:")
        if not os.path.exists(save_dir+'/spectrum/'):
            os.makedirs(save_dir+'/spectrum/')
        
        p1 = plt.plot(x, singular_value,'r-')
        plt.xlabel(u'Log of singular values')
        plt.ylabel(u'Singular Value Rank Index')
        plt.title('singluar value spectrum of embedding, epoch: '+str(epochs))
        # if os.path.exists(save_dir+'spectrum/'):
        #     print("test if found path: ",os.path.exists(save_dir+'spectrum/'),os.path.exists(save_dir),os.path.exists(save_dir+'/spectrum/'))
        #     print("saving spectrum to: ",save_dir+'spectrum/spectrum'+str(epochs)+'.jpg')
        #     plt.savefig(save_dir+'spectrum/spectrum_'+str(epochs)+'.jpg',dpi=300)
        # else:
        #     print("test if found path: ",os.path.exists(save_dir+'spectrum/'),os.path.exists(save_dir),os.path.exists(save_dir+'/spectrum/'))
        #     os.makedirs(save_dir+'spectrum/')
        #     print("test if found path: ",os.path.exists(save_dir+'spectrum/'),os.path.exists(save_dir),os.path.exists(save_dir+'/spectrum/'))
        #     print("create dir and saving spectrum to: ",save_dir+'spectrum/spectrum'+str(epochs)+'.jpg')
        plt.savefig(save_dir+'/spectrum/spectrum_'+str(epochs)+'.jpg',dpi=300)

    
    def LogME_basic(self,embeddings,labels):
        
        return LogME_score(embeddings,labels)

    def SFDA_score(self,embeddings,labels):
        
        try:
            return SFDA_Score(embeddings,labels)
        except:
            print(embeddings)
            print(labels)
    
    def NLEEP_score(self,embeddings,labels):

        return NLEEP(embeddings,labels)
    
    def singular(self, embedding):

        norm = np.linalg.norm(embedding, 2,axis=1)

        z = embedding/(norm[:,np.newaxis] + 0.001)

        # calculate covariance
        # z = z.cpu().detach().numpy()
        z = np.transpose(z)
        c = np.cov(z)

        try: 
            _, d, _ = np.linalg.svd(c)
        except:
            print(c)

        return np.log(d)