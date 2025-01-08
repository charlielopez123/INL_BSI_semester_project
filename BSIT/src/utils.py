from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import linalg
import scipy.sparse as sp

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, device,alpha=[0.05, 0.2, 0.1, 0.3, 0.15, 0.15], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        self.alpha = self.alpha.to(self.device)

    def forward(self, pred, target):

        # print(target)
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

def build_finetune_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
        model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate


    return model_new

def build_finetune_lstm_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to LSTM model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.lstm = model_pretrained.encoder


    return model_new

def build_finetune_gts_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
        model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate
    model_new.conv1 = model_pretrained.conv1
    model_new.conv2 = model_pretrained.conv2
    model_new.fc = model_pretrained.fc
    model_new.fc_out = model_pretrained.fc_out
    model_new.fc_cat = model_pretrained.fc_cat
    model_new.weight_mt = model_pretrained.weight_mt


    return model_new


def build_finetune_gts_simple_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
        model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate
    # model_new.conv1 = model_pretrained.conv1
    # model_new.conv2 = model_pretrained.conv2
    # model_new.fc = model_pretrained.fc
    # model_new.fc_out = model_pretrained.fc_out
    # model_new.fc_cat = model_pretrained.fc_cat
    # model_new.weight_mt = model_pretrained.weight_mt
    return model_new

def build_finetune_bsiformervq(model_new, model_pretrained):
    """
    Load pretrained weights to BSIformervq model
    """
    # Load in pre-trained parameters

    # model_new.conv1 = model_pretrained.conv1
    # model_new.conv2 = model_pretrained.conv2
    # model_new.fc = model_pretrained.fc
    # model_new.fc_out = model_pretrained.fc_out
    # model_new.fc_cat = model_pretrained.fc_cat
    # model_new.weight_mt = model_pretrained.weight_mt
    model_new.encoder = model_pretrained.encoder
    model_new.quantize = model_pretrained.quantize

def build_finetune_bsiformer(model_new, model_pretrained):
    """
    Load pretrained weights to BSIformervq model
    """
    # Load in pre-trained parameters

    model_new.model.input_layer = model_pretrained.model.input_layer
    model_new.model.transformer = model_pretrained.model.transformer
    model_new.model.channel_embedding = model_pretrained.model.channel_embedding
    model_new.model.channel_tokens = model_pretrained.model.channel_tokens


    return model_new

def build_finetune_bsiformerrec(model_new, model_pretrained):
    """
    Load pretrained weights to BSIformervq model
    """
    # Load in pre-trained parameters

    # model_new.conv1 = model_pretrained.conv1
    # model_new.conv2 = model_pretrained.conv2
    # model_new.fc = model_pretrained.fc
    # model_new.fc_out = model_pretrained.fc_out
    # model_new.fc_cat = model_pretrained.fc_cat
    # model_new.weight_mt = model_pretrained.weight_mt
    model_new.model = model_pretrained.model


    return model_new

def build_finetune_gts_only_graph_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    # for l in range(num_rnn_layers):
    #     model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
    #     model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate
    model_new.conv1 = model_pretrained.conv1
    model_new.conv2 = model_pretrained.conv2
    model_new.fc = model_pretrained.fc
    model_new.fc_out = model_pretrained.fc_out
    model_new.fc_cat = model_pretrained.fc_cat
    model_new.weight_mt = model_pretrained.weight_mt


    return model_new

def build_finetune_gts_prob_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
        model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate
    model_new.prob_mt = model_pretrained.prob_mt
    model_new.weight_mt = model_pretrained.weight_mt


    return model_new


def build_finetune_gts_dynamic_model(model_new, model_pretrained, num_rnn_layers,
                         num_layers_frozen=0):
    """
    Load pretrained weights to DCRNN model
    """
    # Load in pre-trained parameters
    for l in range(num_rnn_layers):
        model_new.encoder.encoding_cells[l].dconv_gate = model_pretrained.encoder.encoding_cells[l].dconv_gate
        model_new.encoder.encoding_cells[l].dconv_candidate = model_pretrained.encoder.encoding_cells[l].dconv_candidate
    model_new.conv1 = model_pretrained.conv1
    model_new.conv1w = model_pretrained.conv1w
    model_new.conv2 = model_pretrained.conv2
    model_new.conv2w = model_pretrained.conv2w
    model_new.bn1 = model_pretrained.bn1
    model_new.bn2 = model_pretrained.bn2
    model_new.bn3 = model_pretrained.bn3
    model_new.bn1w = model_pretrained.bn1w
    model_new.bn2w = model_pretrained.bn2w
    model_new.bn3w = model_pretrained.bn3w
    model_new.conv2 = model_pretrained.conv2
    model_new.fc = model_pretrained.fc
    model_new.fcw = model_pretrained.fcw
    model_new.fc_out = model_pretrained.fc_out
    model_new.fc_cat = model_pretrained.fc_cat
    model_new.fc_outw = model_pretrained.fc_outw
    model_new.fc_catw = model_pretrained.fc_catw



    return model_new

def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output

def eval_dict(y_pred, y, y_prob=None, average='weighted'):
    """
    Args:
        y_pred: Predicted labels of all samples
        y : True labels of all samples
        average: 'weighted', 'micro', 'macro' etc. to compute F1 score etc.
    Returns:
        scores_dict: Dictionary containing scores such as F1, acc etc.
        pred_dict: Dictionary containing predictions
        true_dict: Dictionary containing labels
    """

    scores_dict = {}
    pred_dict = defaultdict(list)
    true_dict = defaultdict(list)


    if y is not None:
        scores_dict['acc'] = accuracy_score(y_true=y, y_pred=y_pred)
        scores_dict['F1'] = f1_score(y_true=y, y_pred=y_pred, average=average)
        scores_dict['precision'] = precision_score(
            y_true=y, y_pred=y_pred, average=average)
        scores_dict['recall'] = recall_score(
            y_true=y, y_pred=y_pred, average=average)
        if y_prob is not None:
            if len(np.unique(y)) <= 2:  # binary case
                scores_dict['auroc'] = roc_auc_score(y_true=y, y_score=y_prob)

    return scores_dict, pred_dict, true_dict

def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    y_true_set = np.unique(y_true)
    if len(set(y_true_set)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
            (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh

def mae_loss(y_pred, y_true):

    loss = torch.abs(y_pred - y_true)
    return loss.mean()

def mse_loss(y_pred, y_true):

    loss = (y_pred - y_true).pow(2)
    loss = torch.sqrt(torch.mean(loss))
    return loss

def compute_regression_loss(
        y_true,
        y_predicted,
        standard_scaler=None,
        device=None,
        loss_fn='mae',
        is_tensor=True):
    """
    Compute masked MAE loss with inverse scaled y_true and y_predict
    Args:
        y_true: ground truth signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        y_predicted: predicted signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        standard_scaler: class StandardScaler object
        device: device
        mask: int, masked node ID
        loss_fn: 'mae' or 'mse'
        is_tensor: whether y_true and y_predicted are PyTorch tensor
    """
    if device is not None:
        y_true = y_true.to(device)
        y_predicted = y_predicted.to(device)

    if standard_scaler is not None:
        mean = standard_scaler.mean_[0]
        var = standard_scaler.var_[0]
        y_true = y_true *var+mean
        y_predicted = y_predicted*var+mean

        # print("two shapes: ",y_true.shape,y_predicted.shape)


    if loss_fn == 'mae':
        return mae_loss(y_predicted, y_true)
    else:
        return mse_loss(y_predicted, y_true)

def compute_sampling_threshold(cl_decay_steps, global_step):
    """
    Compute scheduled sampling threshold
    """
    return cl_decay_steps / \
        (cl_decay_steps + np.exp(global_step / cl_decay_steps))

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    State transition matrix D_o^-1W in paper.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    """
    Reverse state transition matrix D_i^-1W^T in paper.
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))