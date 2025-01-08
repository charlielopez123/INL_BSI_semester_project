import torch.nn as nn
from models.attention import AttentionBlock
import torch
import geotorch
import math
import numpy as np
import torchaudio.transforms as T

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
    

class  ConvBSIblock(nn.Module):
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
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, num_patches+1, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(num_patches // self.cut)
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim*self.cut))
        self.init_norm = nn.LayerNorm(num_patches // self.cut)
        self.channel_embedding = nn.Parameter(torch.randn(1, args.num_patches // self.cut, 1))

        self.feature_num = embed_dim
        self.class_num = num_classes

        # self.margin = torch.log(_cls_num_list / torch.sum(_cls_num_list)).cuda()

        # self.conv_filter = nn.Conv1d(num_patches, num_patches, kernel_size=args.kernel_size,stride = args.stride)
        self.period_len = 3
        self.conv_filter1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.conv_filter2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.spectrogram_transform = T.Spectrogram(
            n_fft=200,
            win_length=200,
            hop_length=50,
            center=True,
            window_fn=lambda x: torch.hann_window(x),
        )


        # self.conv_project = nn.Linear(293,num_t_pints)

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
        
        self.projector = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(embed_dim, embed_dim, bias=False),
                                        nn.BatchNorm1d(embed_dim, affine=False)) # output layer

    
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
        
        # x = torch.log( torch.abs( torch.fft.fft(x , dim=-1) ) + 1e-10 )
        # print(x.shape)
        # print("check2:",torch.isneginf(x).any(),x.dtype)

        # x_spec = self.spectrogram_transform(x)

        # # print(x_spec.shape)

        # B,C,F,T = x_spec.shape

        # x = x_spec.reshape((B,C, F*T))
        # print("spectro: ",x_spec.shape)

        # B,T,D = x.shape
        # x = x.reshape((B,T//self.cut,-1))

        _,_,L = x.shape

        x = torch.cat( [torch.log(torch.abs( torch.fft.fft(x , dim=-1).imag[:,:,0:L//2])+1e-8)  , torch.log(torch.abs(torch.fft.fft(x , dim=-1).real[:,:,0:L//2])+1e-8) ] , dim=-1)
        
        # x = x.reshape((B,T,D))
        xshape = x.shape

        x_flat = x.view(-1, xshape[1])  

        output_flat = self.init_norm(x_flat)

        x = output_flat.reshape(xshape)



        B,T,D = x.shape

        # x = x.reshape((B,1,-1))

        # x = self.conv_filter1(x) + x

        # # x = self.conv_filter2(x1) + x1 + x

        # # x = self.conv_filter(x2) + x2

        # x = x.reshape((B,T,D))

        # print(x.shape)

        # x = self.conv_project(x)





        # print(x.mean(dim=1 , keepdim=True))

        # # x = self.init_norm(x)
        
        # fft = torch.fft.fft(x , dim=-1)
        # # print(x)
        # # print(fft)

        
        # print("check3:",torch.isneginf(x).any())
        # x = (x - x.mean(dim=1 , keepdim=True) )/x.std(dim=1 , keepdim=True) 
        # print(x)
        B, T, D = x.shape
        x = self.input_layer(x)
        embed_shape = x.shape

        

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        # print(x.shape)
        x_flat = x.reshape((x.shape[0],x.shape[-1]*self.cut,-1))

        # for i in range(x_flat.shape[-1]):
        #     channel_token_emb_i = (
        #             self.channel_tokens(self.index[i])
        #             .unsqueeze(0)
        #             .repeat(B, self.cut)
        #         )
        #     x_flat[:,:,i] =  x_flat[:,:,i] + channel_token_emb_i

        x_flat = self.positional_encoding(x_flat)

        # x_flat = x_flat + self.pos_embedding
        # x_flat = x.reshape((x.shape[0],-1,x.shape[-1]*self.cut))
        # x_flat = x_flat + self.channel_embedding

        # print(x_flat.shape)

        x = x_flat.reshape(embed_shape)

        # x = torch.cat([cls_token, x], dim=1)
        # x = x + self.pos_embedding[:, : T + 1]

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

        # x = self.projector(x)

        

        # Perform classification prediction
        # cls = x[0]
        # out = self.mlp_head(x.reshape(B,-1))
        return x