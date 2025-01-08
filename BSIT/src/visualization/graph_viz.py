import numpy as np
import os
import sys
import pickle
import networkx as nx
import collections
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import seaborn as sns
import pandas as pd

def get_spectral_graph_positions(file):
    """
    Get positions of EEG electrodes for visualizations
    """

    with open(file, 'rb') as f:
        adj_mx_all = pickle.load(f)
    adj_mx = adj_mx_all[-1]

    node_id_dict = adj_mx_all[1]

    eeg_viz = nx.Graph()
    adj_mx = adj_mx_all[-1]
    node_id_label = collections.defaultdict()

    for i in range(adj_mx.shape[0]):
        eeg_viz.add_node(i)

    for k, v in node_id_dict.items():
        node_id_label[v] = k
    # Add edges
    for i in range(adj_mx.shape[0]):
        for j in range(
                adj_mx.shape[1]):  # do no include self-edge in visualization
            if i != j and adj_mx[i, j] > 0:
                eeg_viz.add_edge(i, j)

    pos = nx.spectral_layout(eeg_viz)
    # keep the nice shape of the electronodes on the scalp
    pos_spec = {node: (y, -x) for (node, (x, y)) in pos.items()}

    return pos_spec

def draw_graph_chord_edge(
        adj_mx,
        node_id_dict,
        pos_spec,
        is_directed,
        title='',
        save_dir=None,
        fig_size=(
            28,
            28),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a graph with weighted edges
    Args:
        adj_mx: Adjacency matrix for the graph, shape (num_nodes, num_nodes
        node_id_dict: dict, key is node name, value is node index
        pos_spec: Graph node position specs from function get_spectral_graph_positions
        is_directed: If True, draw directed graphs
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """

    eeg_viz = nx.DiGraph()
    node_id_label = collections.defaultdict()

    for i in range(adj_mx.shape[0]):
        eeg_viz.add_node(i)

    for k, v in node_id_dict.items():
        node_id_label[v] = k
    
    pos = nx.circular_layout(eeg_viz)

    
    cmap = plt.cm.get_cmap('RdBu')

    node_colors = [cmap(i / 18.0)[:3] for i in range(19)]


    edge_colors = []

    # Add edges
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):  # since it's now directed
            if i != j and adj_mx[i, j] > 0:
                if abs(i - j) > 5:
                    eeg_viz.add_edge(i, j, weight=adj_mx[i, j])
                    edge_colors.append(node_colors[i])

    edges, weights = zip(*nx.get_edge_attributes(eeg_viz, 'weight').items())
    # print(edges)
    def rescale(l,newmin,newmax,rnd=False):
        arr = list(l)
        return [round((x-min(arr))/(max(arr)-min(arr))*(newmax-newmin)+newmin,2) for x in arr]

    weights = rescale(weights,1,35)

    # Change the color scales below

    
    plt.figure(figsize=fig_size)
    # nx.draw_networkx(eeg_viz, pos_spec, labels=node_id_label, with_labels=True,
    #                  edgelist=edges, edge_color=rankdata(weights),
    #                  width=fig_size[1] / 2, edge_cmap=cmap, font_weight='bold',
    #                  node_color=node_color,
    #                  node_size=250 * (fig_size[0] + fig_size[1]),
    #                  font_color='white',
    #                  font_size=font_size)
    nx.draw_networkx_edges(eeg_viz, pos, alpha=0.65, width=weights, edge_color=edge_colors,
                       connectionstyle="arc3,rad=0.24")
    
    nx.draw_networkx_nodes(eeg_viz, pos, node_size=3200, node_color=node_colors, alpha=0.8)

    nx.draw_networkx_labels(eeg_viz, pos, node_id_label,font_size=18, font_color='black')
    plt.title(title, fontsize=font_size)
    plt.axis('equal')
    # if plot_colorbar:
    #     sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    #     sm.set_array([])
    #     plt.colorbar(sm)
    plt.tight_layout()
    if save_dir is not None:
        print("saving to: ",save_dir+'/chord_graph.jpg')
        plt.savefig(save_dir+'/chord_graph.jpg', dpi=300)


def draw_heat_graph(
        p_mx,
        w_mx,
        node_id_dict,
        title='',
        save_dir=None,
        fig_size=(
            12,
            10),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a graph with weighted edges
    Args:
        adj_mx: Adjacency matrix for the graph, shape (num_nodes, num_nodes
        node_id_dict: dict, key is node name, value is node index
        pos_spec: Graph node position specs from function get_spectral_graph_positions
        is_directed: If True, draw directed graphs
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    node_label = []
    for k, v in node_id_dict.items():

        node_label.append(k)

    p_mx = p_mx[:,0].reshape(len(node_label),-1)
    p_mx = 1/(1+1/(np.exp(p_mx)))


    fig = plt.figure(figsize=fig_size)

    sns.heatmap(pd.DataFrame(np.round(p_mx,2),columns = node_label,index = node_label),annot=False, vmax = p_mx.max(),vmin = p_mx.min(),xticklabels = True,yticklabels = True,square=True)
    plt.title("probability graph", fontsize=font_size)


    if save_dir is not None:
        print("saving to: ",save_dir+'/P_graph.jpg')
        plt.savefig(save_dir+'/P_graph.jpg', dpi=300)
    
    fig2 = plt.figure(figsize=fig_size)

    p_mx_onehot = np.where(p_mx>0.05,1,p_mx)
    print("p_mx_onehot:")
    print(p_mx)
    print(p_mx_onehot)

    w_mx = w_mx*p_mx_onehot

    sns.heatmap(pd.DataFrame(np.round(w_mx,2),columns = node_label,index = node_label),annot=False, vmax = w_mx.max(),vmin = 0,xticklabels = True,yticklabels = True,square=True)
    plt.title("weights graph", fontsize=font_size)


    if save_dir is not None:
        print("saving to: ",save_dir+'/W_graph.jpg')
        plt.savefig(save_dir+'/W_graph.jpg', dpi=300)

def draw_disheat_graph(
        p_mx,
        w_mx,
        node_id_dict,
        title='',
        save_dir=None,
        fig_size=(
            12,
            10),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a graph with weighted edges
    Args:
        adj_mx: Adjacency matrix for the graph, shape (num_nodes, num_nodes
        node_id_dict: dict, key is node name, value is node index
        pos_spec: Graph node position specs from function get_spectral_graph_positions
        is_directed: If True, draw directed graphs
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """
    node_label = []
    for k, v in node_id_dict.items():
        node_label.append(k)

    fig = plt.figure(figsize=fig_size)

    sns.heatmap(pd.DataFrame(np.round(p_mx,2),columns = node_label,index = node_label),annot=False, vmax = p_mx.max(),vmin = p_mx.min(),xticklabels = True,yticklabels = True,square=True)
    plt.title("probability graph", fontsize=font_size)


    if save_dir is not None:
        print("saving to: ",save_dir+'/P_graph.jpg')
        plt.savefig(save_dir+'/P_graph.jpg', dpi=300)
    
    fig2 = plt.figure(figsize=fig_size)

    sns.heatmap(pd.DataFrame(np.round(w_mx,2),columns = node_label,index = node_label),annot=False, vmax = 1.7,vmin = 0,xticklabels = True,yticklabels = True,square=True)
    plt.title("weights graph", fontsize=font_size)


    if save_dir is not None:
        print("saving to: ",save_dir+'/W_graph_scaled.jpg')
        plt.savefig(save_dir+'/W_graph.jpg', dpi=300)



def draw_graph_weighted_edge(
        adj_mx,
        node_id_dict,
        pos_spec,
        is_directed,
        title='',
        save_dir=None,
        fig_size=(
            12,
            8),
    node_color='Red',
    font_size=20,
        plot_colorbar=False):
    """
    Draw a graph with weighted edges
    Args:
        adj_mx: Adjacency matrix for the graph, shape (num_nodes, num_nodes
        node_id_dict: dict, key is node name, value is node index
        pos_spec: Graph node position specs from function get_spectral_graph_positions
        is_directed: If True, draw directed graphs
        title: str, title of the figure
        save_dir: Dir to save the plot
        fig_size: figure size

    """

    eeg_viz = nx.DiGraph() if is_directed else nx.Graph()
    node_id_label = collections.defaultdict()

    for i in range(adj_mx.shape[0]):
        eeg_viz.add_node(i)

    for k, v in node_id_dict.items():
        node_id_label[v] = k

    # Add edges
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):  # since it's now directed
            if i != j and adj_mx[i, j] > 0:
                eeg_viz.add_edge(i, j, weight=adj_mx[i, j])

    edges, weights = zip(*nx.get_edge_attributes(eeg_viz, 'weight').items())

    # Change the color scales below
    k = 3
    cmap = plt.cm.Greys(np.linspace(0, 1, (k + 1) * len(weights)))
    cmap = matplotlib.colors.ListedColormap(cmap[len(weights):-1:(k - 1)])

    plt.figure(figsize=fig_size)
    nx.draw_networkx(eeg_viz, pos_spec, labels=node_id_label, with_labels=True,
                     edgelist=edges, edge_color=rankdata(weights),
                     width=fig_size[1] / 2, edge_cmap=cmap, font_weight='bold',
                     node_color=node_color,
                     node_size=250 * (fig_size[0] + fig_size[1]),
                     font_color='white',
                     font_size=font_size)
    plt.title(title, fontsize=font_size)
    plt.axis('off')
    if plot_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(
                vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm)
    plt.tight_layout()
    if save_dir is not None:
        print("saving to: ",save_dir+'/graph.jpg')
        plt.savefig(save_dir+'/graph.jpg', dpi=300)

