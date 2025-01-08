from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.BSI_layers_vq import BSIblock
from models.BSI_layers_pred import BSIblock_pred
from models.ConvBSI_layers import ConvBSIblock
from models.norm_ema_quantizer import NormEMAVectorQuantizer
import torch.nn.functional as F
# import lightning as L
from einops import rearrange
class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()
        # self.modelname = model_kwargs["model_name"]

        self.encoder = BSIblock(args, True)
        self.decoder = BSIblock(args, False)
        self.args = args

        self.quantize = NormEMAVectorQuantizer(
            n_embed=args.n_embed, embedding_dim=args.embed_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.embedding_dim = args.embed_dim
        self.embedding_predict_dim = self.embedding_dim // 2
        self.embedding_norm = nn.LayerNorm(args.embed_dim)
        self._device = torch.device('{}'.format(args.gpu))
        self.decode_task_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, args.num_t_pints),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, args.num_t_pints),
        )
        embed_dim = args.embed_dim
        num_classes = args.n_classes
        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(embed_dim, num_classes),
                                        nn.BatchNorm1d(num_classes, affine=False)) # output layer
        # self.predictor.require_grad = False

        self.criterion = nn.BCEWithLogitsLoss()
        # self.learning_rate = model_kwargs["learning_rate"]
        # self.wd = model_kwargs["weight_decay"]
        self.history_embedding = torch.zeros(args.n_classes, args.embed_dim).to(self._device)
        print("embed_dim: ",args.embed_dim)
        self.history_embedding.requires_grad = False

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        loss_fn = F.mse_loss
        rec_loss = loss_fn(rec, target)
        return rec_loss


    def encode(self, x):


        batch_size, n, a, t = x.shape
        x = torch.cat( [torch.log(torch.abs( torch.fft.fft(x , dim=-1).imag[:,:,:,0:t//2])+1e-8)  , torch.log(torch.abs(torch.fft.fft(x , dim=-1).real[:,:,:,0:t//2])+1e-8) ] , dim=-1)
        encoder_features = self.encoder.forward_feature(x)

        # with torch.cuda.amp.autocast(enabled=False):
        #     to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = encoder_features.shape[1]
        h, w = n, N // n

        encoder_features = rearrange(encoder_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(encoder_features)

        return quantize, embed_ind, loss
    
    def decode(self, quantize):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        decoder_features = self.decoder.forward_feature(quantize)
        rec = self.decode_task_layer(decoder_features)
        rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_angle
    

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def forward_tokenize(self, x):
        """
        x: shape [B, N, T]
        """

        B,T,D = x.shape

        x = x.reshape((B,-1,self.args.num_cut,D))

        # x = rearrange(x, 'B N (A T) -> B N A T', T=)
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)

        quantize, embed_ind, emb_loss = self.encode(x)

        # print("tokens: ")
        # print(embed_ind)
        
        xrec, xrec_angle = self.decode(quantize)
        rec_loss = self.calculate_rec_loss(xrec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)
        loss = emb_loss + rec_loss + rec_angle_loss

        return loss
    
    def forward(self,x, train = False):
        B,T,D = x.shape

        x = x.reshape((B,-1,self.args.num_cut,D))

        quantize, embed_ind, emb_loss = self.encode(x)

        quantize = rearrange(quantize, 'b c h w -> b (h w) c') # reshape for quantizer

        quantize = quantize.mean(dim = 1)

        quantize = self.embedding_norm(quantize)

        logit = self.mlp_head(quantize)

        return  logit, quantize.cpu().detach().numpy()

    # def forward(self, x, train = False):

    #     return self.model(x, train)

