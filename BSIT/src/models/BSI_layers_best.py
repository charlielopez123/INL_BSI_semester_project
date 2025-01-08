import torch.nn as nn
from models.attention import AttentionBlock
import torch
import geotorch
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # Exclude the last term to match the shape
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        # print(x.shape,self.pe[:, : x.size(1)].shape)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

class BSIblock(nn.Module):
    def __init__(
        self,
        args
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        embed_dim = args.embed_dim
        hidden_dim = args.hidden_dim
        num_t_pints = args.num_t_pints
        num_heads = args.num_heads
        num_layers = args.num_layers
        num_classes = args.n_classes
        num_patches = args.num_patches
        dropout = args.dropout
        n_channels = args.num_patches // args.num_cut
        self.args = args
        self.etf = args.etf
        self.task = args.task
        self.cut = args.num_cut
        self.input_dim = num_t_pints
        self.input_layer = nn.Linear(num_t_pints  , embed_dim)
        # self.input_layer = nn.Sequential(nn.Linear(num_t_pints, hidden_dim),
        #                         nn.ReLU(inplace=True), # first layer
        #                         nn.Linear(hidden_dim, embed_dim))


        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, num_patches+1, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(num_patches // self.cut)


        self.init_norm = nn.LayerNorm(num_patches)

        self.feature_num = embed_dim
        self.class_num = num_classes

        self.margin = []

        self.channel_tokens = nn.Embedding(n_channels, embed_dim)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )
        

        if args.linear_probing == 1:
            for p in self.parameters():
                p.requires_grad = False

        if self.feature_num < self.class_num:
            self.rotate = nn.Linear(self.class_num, self.feature_num, bias=False)
            self.register_buffer("ETF", self.generate_ETF(dim=self.class_num))
        else:
            self.rotate = nn.Linear(self.feature_num, self.feature_num, bias=False)
            self.register_buffer("ETF", \
                self.generate_ETF(dim=self.feature_num)[:, :self.class_num])
        geotorch.orthogonal(self.rotate, "weight")

        self.mlp_head = nn.Linear(embed_dim, num_classes)

        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.mlp_head,
                                        nn.BatchNorm1d(num_classes, affine=False)) # output layer


    
    def generate_ETF(self, dim):
        return torch.eye(dim, dim) - torch.ones(dim, dim) / dim

    def forward(self, x, train = False):
        # print(self.forward_feature(x))
        # print(x.shape)
        feature = self.forward_feature(x, train)
        if self.etf:
            logit = feature @ self.rotate.weight @ self.ETF
        else:
            logit = self.mlp_head(feature)
        if self.task != "SSLEval":
            feature = feature.cpu().detach().numpy()
        # return logit, feature
        # print(self.margin)
        # if not train:
        #     logit = logit - self.margin
            
        return logit, feature

    def get_classweight(self):
        return (self.rotate.weight @ self.ETF).T


    def forward_feature(self, x, train =False):
        # Preprocess input
        

        x = torch.cat( [torch.log(torch.abs( torch.fft.fft(x , dim=-1).imag[:,:,0:self.input_dim//2])+1e-8)  , torch.log(torch.abs(torch.fft.fft(x , dim=-1).real[:,:,0:self.input_dim//2])+1e-8) ] , dim=-1)

        xshape = x.shape

        x_flat = x.view(-1, xshape[1])  

        output_flat = self.init_norm(x_flat)

        x = output_flat.reshape(xshape)

        B, T, D = x.shape
        x = self.input_layer(x)
        embed_shape = x.shape


        x_flat = x.reshape((x.shape[0],x.shape[-1]*self.cut,-1))

        x_flat = self.positional_encoding(x_flat)

        x = x_flat.reshape(embed_shape)

        # Apply Transforrmer
        # x = self.dropout(x)
        if train and self.args.mask_rate > 0:

            random_int = np.random.randint(int((1 - self.args.mask_rate) * x.shape[1]), x.shape[1]+int((self.args.mask_rate) * x.shape[1]))

            indices = torch.randperm(x.shape[1])[:random_int]

            x = x[:,indices,:]

        x = x.transpose(0, 1)
        x = self.transformer(x)

        x = x.transpose(0,1).mean(dim=1)

        x = self.embedding_norm(x)

        return x