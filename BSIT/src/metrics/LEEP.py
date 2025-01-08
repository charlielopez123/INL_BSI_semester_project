import numpy as np
import torch

def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

def LEEP(X, y, model_name='resnet50'):

    n = len(y)
    num_classes = len(np.unique(y))

    # read classifier
    # Group1: model_name, fc_name, model_ckpt
    ckpt_models = {
        'densenet121': ['classifier.weight', './models/group1/checkpoints/densenet121-a639ec97.pth'],
        'densenet169': ['classifier.weight', './models/group1/checkpoints/densenet169-b2777c0a.pth'],
        'densenet201': ['classifier.weight', './models/group1/checkpoints/densenet201-c1103571.pth'],
        'resnet34': ['fc.weight', './models/group1/checkpoints/resnet34-333f7ec4.pth'],
        'resnet50': ['fc.weight', './models/group1/checkpoints/resnet50-19c8e357.pth'],
        'resnet101': ['fc.weight', './models/group1/checkpoints/resnet101-5d3b4d8f.pth'],
        'resnet152': ['fc.weight', './models/group1/checkpoints/resnet152-b121ed2d.pth'],
        'mnasnet1_0': ['classifier.1.weight', './models/group1/checkpoints/mnasnet1.0_top1_73.512-f206786ef8.pth'],
        'mobilenet_v2': ['classifier.1.weight', './models/group1/checkpoints/mobilenet_v2-b0353104.pth'],
        'googlenet': ['fc.weight', './models/group1/checkpoints/googlenet-1378be20.pth'],
        'inception_v3': ['fc.weight', './models/group1/checkpoints/inception_v3_google-1a9a5a14.pth'],
    }
    ckpt_loc = ckpt_models[model_name][1]
    fc_weight = ckpt_models[model_name][0]
    fc_bias = fc_weight.replace('weight', 'bias')
    ckpt = torch.load(ckpt_loc, map_location='cpu')
    fc_weight = ckpt[fc_weight].detach().numpy()
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)   # p(z|x), N x C(source)

    pyz = np.zeros((num_classes, 1000))  # C(source) = 1000
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n
    
    pz = np.sum(pyz, axis=0)     # marginal probability
    py_z = pyz / pz              # conditional probability, C x C(source)
    py_x = np.dot(prob, py_z.T)  # N x C

    # leep = E[p(y|x)]
    leep_score = np.sum(py_x[np.arange(n), y]) / n
    return leep_score