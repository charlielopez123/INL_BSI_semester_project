from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.BSI_layers import BSIblock
from models.BSI_layers_pred import BSIblock_pred
from models.BSI_layersT import BSITblock
from models.BSI_layers_fft import BSIblockfft
from models.ConvBSI_layers import ConvBSIblock
# import lightning as L

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()
        # self.modelname = model_kwargs["model_name"]
        if args.task == "Regression":
            self.model = BSIblock_pred(args)
        elif args.model == "BSIformerT":
            self.model = BSITblock(args)
        elif args.dataset == "BSIsamplewavelet":
            self.model = BSIblock(args)
        else:
            self.model = BSIblockfft(args)
        if args.model == "ConvBSIformer":
            self.model = ConvBSIblock(args)
        # self.aug_variance = model_kwargs["aug_variance"]
        self.embedding_dim = args.embed_dim
        self.embedding_predict_dim = self.embedding_dim // 2
        self._device = torch.device('{}'.format(args.gpu))
        # self.pre_training = model_kwargs["pre_training"]
        # self.save_hyperparameters()
        # self.pretraining_method = model_kwargs["pretraining_method"]
        # # print(self.device)
        # self.device_ = model_kwargs["gpu"]
        # self.supcon_loss = SupConLoss(model_kwargs["gpu"])

        self.predictor = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_predict_dim,bias=False),
                                nn.BatchNorm1d(self.embedding_predict_dim),
                                nn.ReLU(inplace=True), # hidden layer
                                nn.Linear(self.embedding_predict_dim, self.embedding_dim)) # output layer
        
        # self.predictor.require_grad = False

        self.criterion = nn.BCEWithLogitsLoss()
        # self.learning_rate = model_kwargs["learning_rate"]
        # self.wd = model_kwargs["weight_decay"]
        self.history_embedding = torch.zeros(args.n_classes, args.embed_dim).to(self._device)
        print("embed_dim: ",args.embed_dim)
        self.history_embedding.requires_grad = False

    def forward(self, x, train = False):

        return self.model(x, train)

