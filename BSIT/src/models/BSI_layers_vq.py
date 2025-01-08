import torch.nn as nn
from models.attention import AttentionBlock
import torch
import geotorch
import math
import numpy as np
from einops import rearrange



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


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = 62 * (EEG_size // patch_size)
        # self.patch_shape = (1, EEG_size // patch_size)
        # self.EEG_size = EEG_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        # print("after patch proj: ",x.shape)
        return x


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.ReLU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.ReLU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.ReLU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        # print("after temporal conv: ",x.shape)
        return x
    

class BSIblock(nn.Module):
    def __init__(
        self,
        args,
        encoder = True
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
        
        # self.input_layer = nn.Sequential(nn.Linear(num_t_pints, hidden_dim),
        #                         nn.ReLU(inplace=True), # first layer
        #                         nn.Linear(hidden_dim, embed_dim))
        self.encoder = encoder
        if self.encoder:
            self.patch_embed = TemporalConv(out_chans = 8)
            self.input_layer = nn.Linear(1776  , embed_dim)
        else:
            self.patch_embed = PatchEmbed(patch_size = 1, in_chans = embed_dim, embed_dim = embed_dim)
            self.input_layer = nn.Linear(32  , embed_dim)


        self.transformer = nn.Sequential(
            *(AttentionBlock(args, embed_dim, num_patches, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(num_patches // self.cut)

        self.init_norm = nn.LayerNorm(num_patches)

        self.feature_num = embed_dim
        self.class_num = num_classes

        # self.margin = torch.log(_cls_num_list / torch.sum(_cls_num_list)).cuda()

        self.margin = []

        self.channel_tokens = nn.Embedding(n_channels, embed_dim)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )
        

        # if args.linear_probing ==W 1:
        #     for p in self.parameters():
        #         p.requires_grad = False

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

        # self.projector = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False),
        #                                 nn.BatchNorm1d(embed_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Linear(embed_dim, embed_dim, bias=False),
        #                                 nn.BatchNorm1d(embed_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 nn.Linear(embed_dim, embed_dim, bias=False),
        #                                 nn.BatchNorm1d(embed_dim, affine=False)) # output layer
        

    
    def generate_ETF(self, dim):
        return torch.eye(dim, dim) - torch.ones(dim, dim) / dim

    def forward(self, x, train = False):
        # print(self.forward_feature(x))
        # print(x.shape)
        feature = self.forward_feature(x, train)
        if self.etf:
            logit = feature @ self.rotate.weight @ self.ETF
            # if not train:
            #     logit = logit - self.margin
        else:
            logit = self.mlp_head(feature)
        if self.task != "SSLEval":
            feature = feature.cpu().detach().numpy()
        # return logit, feature
        # print(self.margin)

            
        return logit, feature

    def get_classweight(self):
        return (self.rotate.weight @ self.ETF).T


    def forward_feature(self, x, train =False):
        # Preprocess input
        
        # x = torch.log( torch.abs( torch.fft.fft(x , dim=-1) ) + 1e-10 )
        # print(x.shape)
        # print("check2:",torch.isneginf(x).any(),x.dtype)
        # print(x.shape)

        # B,T,_ = x.shape

        # x = x.reshape((B,T//self.cut,-1))

        # B,T,D = x.shape

        # x = x.reshape((B,-1,self.cut,D))

        B,Channel,Patches,D = x.shape

        x = self.patch_embed(x)
        

        # x = x.reshape((B,T,-1))

        xshape = x.shape

        x_flat = x.view(-1, xshape[1])  

        output_flat = self.init_norm(x_flat)

        x = output_flat.reshape(xshape)


        B, T, D = x.shape
        x = self.input_layer(x)
        


        embed_shape = x.shape


        x_flat = x.reshape((x.shape[0],x.shape[-1]*self.cut,-1))

        for i in range(x_flat.shape[-1]):
            channel_token_emb_i = (
                    self.channel_tokens(self.index[i])
                    .unsqueeze(0)
                    .repeat(B, self.cut)
                )
            x_flat[:,:,i] =  x_flat[:,:,i] + channel_token_emb_i

        x_flat = self.positional_encoding(x_flat)

        x = x_flat.reshape(embed_shape)

        
        # x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        # x = self.dropout(x)
        if train and self.args.mask_rate > 0:

            random_int = np.random.randint(int((1 - self.args.mask_rate) * x.shape[1]), x.shape[1]+int((self.args.mask_rate) * x.shape[1]))

            indices = torch.randperm(x.shape[1])[:random_int]

            x = x[:,indices,:]

        # x = x.transpose(0, 1)
        x = self.transformer(x)

        # x = x.mean(dim=1)
        # x = x.transpose(0,1).mean(dim=1)

        # x = x.transpose(0,1)

        x = self.embedding_norm(x)

        # x = self.projector(x)

        

        # Perform classification prediction
        # cls = x[0]
        # out = self.mlp_head(x.reshape(B,-1))
        return x