from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.BSI_layers_dec import BSIblock as BSIblock_rec
from models.BSI_layers import BSIblock
from models.ConvBSI_layers import ConvBSIblock
import torch.nn.functional as F
# import lightning as L

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()
        # self.modelname = model_kwargs["model_name"]

        self.model = BSIblock(args)
        self.model_dec = BSIblock_rec(args)

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
    
    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        x = (x - mean) / std
        return x

    def calculate_rec_loss(self, rec, target):
        loss_fn = F.mse_loss
        rec_loss = loss_fn(rec, target)
        return rec_loss
    
    def forward(self, x, train = False):

        return self.model(x, train)

    def forward_tokenize(self, x, train = False):

        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)


        embedding = self.model.forward_enc(x,train)
        amp_rec, angle_rec = self.model_dec.forward_dec(embedding,train)

        rec_loss = self.calculate_rec_loss(amp_rec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(angle_rec, angle)

        

        return rec_loss + rec_angle_loss

