import numpy as np
import os
import sys
import pickle
import collections
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from metrics.LogME import LogME
from metrics.SFDA import SFDA_Score


def LogME_score(embeddings,labels):
    embedding_ = embeddings.copy()
    labels_ = labels.copy()
    labels_ = labels_.astype(int)
    # print(labels[:10],labels.flatten()[:10])
    unique_elements = np.unique(labels_)
    label_onehot = np.eye(len(unique_elements))[labels_.flatten()]

    logme = LogME(regression=True)
    score = logme.fit(embedding_,label_onehot)
    # print("LogME: ",score)
    return score


def tSNE_vis(
        embeddings,
        labels,
        epochs,
        train_loss,
        save_dir=None,
        ssl=None,
        embedding_dict = None,
        embedding_batch = None,
        fig_size=(
            12,
            8),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a 2D illustration of hidden embeddings
    Args:
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 800, random_state=0)
    tsne_results = tsne.fit_transform(embeddings)

    logme_score = LogME_score(embeddings,labels)
    sfda_score = SFDA_Score(embeddings,labels)

    legend_size = 14
    title_size = 20
    label_size = 20

    
      
    # Plotting
    plt.figure(figsize=fig_size)
    # f, ax = plt.subplots(1)

    unique_elements = np.unique(labels)
    colors = ['y','b','k','r','g','m','c','tab:orange','tab:grey','tab:brown','royalblue','peru','teal','lawngreen']

    for index,labeli in enumerate(unique_elements):


        indices = np.where(labels == labeli)[0]
      
        plt.scatter(tsne_results[indices,0], tsne_results[indices,1], 
                c = colors[index], alpha = 0.9, label = "class "+str(labeli))
    

  
    plt.legend()

    if ssl is not None:
        valid_loss,test_loss = ssl
        plt.title('t-SNE plot epoch: '+str(epochs)+", SSL valid: "+str(format((valid_loss),'.2f'))+", SSL test: "+str(format((test_loss),'.2f'))+", \n LogME: "+str(format((logme_score),'.3f'))+", SFDA: "+str(format((sfda_score),'.3f')),fontsize=title_size)
    elif train_loss is not None:  
        plt.title('t-SNE plot epoch: '+str(epochs)+", train score: "+str(format((train_loss),'.2f'))+", \n LogME: "+str(format((logme_score),'.3f'))+", SFDA: "+str(format((sfda_score),'.3f')),fontsize=title_size)
    else:
        plt.title('t-SNE plot epoch: '+str(epochs)+", \n LogME: "+str(format((logme_score),'.3f'))+", SFDA: "+str(format((sfda_score),'.3f')),fontsize=title_size)

    plt.xticks(fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('x_tsne',fontsize = label_size)
    plt.ylabel('y_tsne',fontsize = label_size)

    if not os.path.exists(save_dir+'/tSNE/'):
        os.makedirs(save_dir+'/tSNE/')

    if save_dir is not None:
        print("saving to: ",save_dir+'tSNE/tSNE_'+str(epochs)+'.jpg')
        plt.savefig(save_dir+'/tSNE/tSNE_'+str(epochs)+'.jpg', dpi=300)

def mux_PCA_vis(
        pca,
        wx,
        labels,
        mu_x,
        epochs,
        save_dir=None,
        fig_size=(
            12,
            8),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a 2D illustration of hidden embeddings
    Args:
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    pca_data = pca.transform(wx)

    # print(paverage.shape)
      
    # Plotting
    plt.figure(figsize=fig_size)
    # f, ax = plt.subplots(1,fig_size=fig_size)

    unique_elements = np.unique(labels)
    colors = ['y','b','k','r','g','m','c','tab:orange','tab:grey','tab:brown','royalblue','peru','teal','lawngreen']

    for index,labeli in enumerate(unique_elements):


        indices = np.where(labels == labeli)[0]
      
        plt.scatter(pca_data[indices,0], pca_data[indices,1], 
                c = colors[index], alpha = 0.9, label = "class "+str(labeli))
    
    pca_mux= pca.transform(mu_x)

    for index,labeli in enumerate(unique_elements):

        x = pca_mux[index,0]
        y = pca_mux[index,1]
        plt.scatter(x, y, s = 300, c = colors[index] ,marker = "P", alpha = 0.95, linewidths= 2, edgecolors = 'w')
    
  
    plt.legend()


    plt.title('Output Space PCA plot epoch: '+str(epochs))

    plt.xlabel('x_PCA')
    plt.ylabel('y_PCA')


    if not os.path.exists(save_dir+'/PCA_Auto/'):
        os.makedirs(save_dir+'/PCA_Auto/')

    if save_dir is not None:
        print("saving to: ",save_dir+'PCA_Auto/PCA_'+str(epochs)+'.jpg')
        plt.savefig(save_dir+'PCA_Auto/PCA_'+str(epochs)+'.jpg', dpi=300)

def PCA_vis(
        pca,
        embeddings,
        labels,
        epochs,
        train_loss,
        save_dir=None,
        ssl=None,
        embedding_dict=None,
        embedding_batch = None,
        fig_size=(
            12,
            8),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a 2D illustration of hidden embeddings
    Args:
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    pca_data = pca.transform(embeddings)
    legend_size = 14
    title_size = 20
    label_size = 20



    # print(paverage.shape)
      
    # Plotting
    plt.figure(figsize=fig_size)
    # f, ax = plt.subplots(1,fig_size=fig_size)

    logme_score = LogME_score(embeddings,labels)
    sfda_score = SFDA_Score(embeddings,labels)

    unique_elements = np.unique(labels)
    colors = ['y','b','k','r','g','m','c','tab:orange','tab:grey','tab:brown','royalblue','peru','teal','lawngreen']

    for index,labeli in enumerate(unique_elements):


        indices = np.where(labels == labeli)[0]
      
        plt.scatter(pca_data[indices,0], pca_data[indices,1], 
                c = colors[index], alpha = 0.9, label = "class "+str(labeli))
    
    if embedding_dict is not None:

        pca_average = pca.transform(embedding_dict)
    
        for index,labeli in enumerate(unique_elements):

            x = pca_average[index,0]
            y = pca_average[index,1]
            plt.scatter(x, y, s = 300, c = colors[index] ,marker = "P", alpha = 0.95, linewidths= 2, edgecolors = 'w')
    
    if embedding_batch is not None:

        for batch in embedding_batch:

            pca_average = pca.transform(batch.cpu().detach().numpy())
    
            for index,labeli in enumerate(unique_elements):

                x = pca_average[index,0]
                y = pca_average[index,1]
                plt.scatter(x, y, s = 30, c = colors[index] ,marker = "X", alpha = 0.65, linewidths= 0.2, edgecolors = 'w')

  
    plt.legend(fontsize=legend_size)

    if ssl is not None:
        valid_loss,test_loss = ssl
        plt.title('PCA plot epoch: '+str(epochs)+", SSL valid: "+str(format((valid_loss),'.2f'))+", SSL test: "+str(format((test_loss),'.2f'))+", \n LogME: "+str(format((logme_score),'.2f'))+", SFDA: "+str(format((sfda_score),'.4f')),fontsize=title_size)
    elif train_loss is not None:
        plt.title('PCA plot epoch: '+str(epochs)+", train score: "+str(format((train_loss),'.2f'))+", \n LogME: "+str(format((logme_score),'.2f'))+", SFDA: "+str(format((sfda_score),'.4f')),fontsize=title_size)
    else:
        plt.title('PCA plot epoch: '+str(epochs)+", \n LogME: "+str(format((logme_score),'.2f'))+", SFDA: "+str(format((sfda_score),'.4f')),fontsize=title_size)

    plt.xticks(fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('x_PCA',fontsize = label_size)
    plt.ylabel('y_PCA',fontsize = label_size)


    if not os.path.exists(save_dir+'/PCA/'):
        os.makedirs(save_dir+'/PCA/')

    if save_dir is not None:
        print("saving to: ",save_dir+'PCA/PCA_'+str(epochs)+'.jpg')
        plt.savefig(save_dir+'PCA/PCA_'+str(epochs)+'.jpg', dpi=300)


def simple_PCA_vis(
        pca,
        embeddings,
        labels,
        save_dir=None,
        title=None,
        fig_size=(
            12,
            8),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a 2D illustration of hidden embeddings
    Args:
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    pca_data = pca.transform(embeddings)
    legend_size = 14
    title_size = 20
    label_size = 20



    # print(paverage.shape)
      
    # Plotting
    plt.figure(figsize=fig_size)
    # f, ax = plt.subplots(1,fig_size=fig_size)

    unique_elements = np.unique(labels)
    colors = ['y','b','k','r','g','m','c','tab:orange','tab:grey','tab:brown','royalblue','peru','teal','lawngreen']

    for index,labeli in enumerate(unique_elements):


        indices = np.where(labels == labeli)[0]
      
        plt.scatter(pca_data[indices,0], pca_data[indices,1], 
                c = colors[index], alpha = 0.9, label = "class "+str(labeli))
    

  
    plt.legend(fontsize=legend_size)

    plt.title(title, fontsize=title_size)
    plt.xticks(fontsize = label_size)
    plt.yticks(fontsize = label_size)
    plt.xlabel('x_PCA',fontsize = label_size)
    plt.ylabel('y_PCA',fontsize = label_size)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_dir+title+'.jpg', dpi=300)

def EEG_PCA_vis(
        pca,
        pca2,
        embeddings,
        labels,
        epochs,
        train_loss,
        save_dir=None,
        ssl=None,
        embedding_dict = None,
        embedding_batch = None,
        fig_size=(
            24,
            8),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a 2D illustration of hidden embeddings
    Args:
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    pca_data = pca.transform(embeddings)
    pca_data2 = pca2.transform(embeddings)

    plt.figure(figsize=(24,8))
      
    # Plotting
    plt.subplot(1, 2, 1)

    logme_score = LogME_score(embeddings,labels)
    sfda_score = SFDA_Score(embeddings,labels)

    unique_elements = np.unique(labels)
    colors = ['y','b','k','r']

    for index,labeli in enumerate(unique_elements):


        indices = np.where(labels == labeli)[0]
      
        plt.scatter(pca_data[indices,0], pca_data[indices,1], 
                c = colors[index], alpha = 0.9, label = "class "+str(labeli))

    if embedding_dict is not None:

        pca_average = pca.transform(embedding_dict)
    
        for index,labeli in enumerate(unique_elements):

            x = pca_average[index,0]
            y = pca_average[index,1]
            plt.scatter(x, y, s = 300, c = colors[index] ,marker = "P", alpha = 0.95, linewidths= 2, edgecolors = 'w')
    
    if embedding_batch is not None:

        for batch in embedding_batch:

            pca_average = pca.transform(batch.cpu().detach().numpy())
    
            for index,labeli in enumerate(unique_elements):

                x = pca_average[index,0]
                y = pca_average[index,1]
                plt.scatter(x, y, s = 30, c = colors[index] ,marker = "X", alpha = 0.65, linewidths= 0.2, edgecolors = 'w')
    
  
    plt.legend(prop = {'size':20})

    plt.xlabel('x_PCA',fontsize=20)
    plt.ylabel('y_PCA',fontsize=20)

    plt.subplot(1, 2, 2)

    elements = [0,2]

    for index,labeli in enumerate(elements):


        indices = np.where(labels == labeli)[0]
      
        plt.scatter(pca_data2[indices,0], pca_data2[indices,1], 
                c = colors[labeli], alpha = 0.9, label = "class "+str(labeli))
    
    plt.legend(prop = {'size':20})

    if ssl is not None:
        valid_loss,test_loss = ssl
        plt.suptitle('PCA plot epoch: '+str(epochs)+", SSL valid: "+str(format((valid_loss),'.2f'))+", SSL test: "+str(format((test_loss),'.2f'))+", LogME: "+str(format((logme_score),'.4f'))+", SFDA: "+str(format((sfda_score),'.4f')),fontsize=22)
    else:
        plt.suptitle('PCA plot epoch: '+str(epochs)+", train F1: "+str(format((train_loss),'.2f'))+", LogME: "+str(format((logme_score),'.4f'))+", SFDA: "+str(format((sfda_score),'.4f')),fontsize=22)
    plt.xlabel('x_PCA',fontsize=20)
    plt.ylabel('y_PCA',fontsize=20)


    if not os.path.exists(save_dir+'/PCA/'):
        os.makedirs(save_dir+'/PCA/')

    if save_dir is not None:
        print("saving to: ",save_dir+'PCA/PCA_'+str(epochs)+'.jpg')
        plt.savefig(save_dir+'PCA/PCA_'+str(epochs)+'.jpg', dpi=300)