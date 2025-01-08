import torch.nn as nn
from models.attention import AttentionBlock, AttentionDecoderBlock
import torch
import geotorch
import math

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
    

class BSIblock_pred(nn.Module):
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
        num_decheads = args.num_decheads
        num_layers = args.num_layers
        num_declayers = args.num_declayers
        num_classes = args.n_classes
        num_patches = args.num_patches
        dropout = args.dropout
        n_channels = args.num_patches // args.num_cut
        self.etf = args.etf
        self.task = args.task
        self.cut = args.num_cut
        self.batch = args.batch_size
        device = torch.device('{}'.format(args.gpu))
        self.input_dim = num_t_pints
        self.input_layer = nn.Linear(num_patches  , embed_dim)
        self.input_layer_dec = nn.Linear(num_patches  , embed_dim)
        # self.input_layer_dec = nn.Linear(5  , 5)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, num_patches+1, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        # self.transformerdecoder = nn.Sequential(
        #     *(AttentionDecoderBlock(embed_dim, num_patches+1, hidden_dim, num_heads, dropout=dropout) for _ in range(num_declayers))
        # )
        self.transformerdecoder = AttentionDecoderBlock(embed_dim, num_patches+1, hidden_dim, num_decheads, dropout=dropout)
        # self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_t_pints*5*self.cut))

        # self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim),nn.Dropout(dropout),nn.Linear(embed_dim, 5))

        self.mlp_head = nn.Linear(embed_dim, 5)

        # self.mlp_head = nn.Linear(embed_dim, num_t_pints*self.cut)
        # self.mlp_channel = nn.Linear(num_patches, 5)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.positional_encoding_dec = PositionalEncoding(embed_dim)
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim*self.cut))
        self.init_norm = nn.LayerNorm(num_patches)
        self.channel_embedding = nn.Parameter(torch.randn(1, args.num_patches // self.cut, 1))

        self.dec_inp = torch.zeros(self.batch,num_t_pints,num_patches).to(device)

        self.feature_num = embed_dim
        self.class_num = num_classes
        # _cls_num_list = torch.Tensor(_cls_num_list)
        # self.margin = torch.log(_cls_num_list / torch.sum(_cls_num_list)).cuda()
        self.channel_tokens = nn.Embedding(n_channels, embed_dim)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def forward(self, x):
        # print(self.forward_feature(x))
        # print(x.shape)
        feature = self.forward_feature(x)

        # pred_channel = self.mlp_channel(feature.transpose(1,2))

        # pred_channel = pred_channel.transpose(1,2)
        # print("pred y: ",pred.shape)
        pred = self.mlp_head(feature)
        # pred = feature
        pred = pred.transpose(1,2)
        # print(pred.shape)
        # pred = pred.reshape((-1,5,self.cut * self.input_dim))
        if self.task != "SSLEval":
            feature = feature.cpu().detach().numpy()
        return pred, feature
        # return logit if self.training else logit - self.margin


    def forward_feature(self, x):
        # Preprocess input
        
        # x = torch.log( torch.abs( torch.fft.fft(x , dim=-1) ) + 1e-10 )
        # print(x.shape)
        # print("check2:",torch.isneginf(x).any(),x.dtype)

        # xshape = x.shape

        # x_flat = x.view(-1, xshape[1])  

        # output_flat = self.init_norm(x_flat)

        # x = output_flat.reshape(xshape)

        B, C, L = x.shape

        # x = torch.cat( [torch.log(torch.abs( torch.fft.fft(x , dim=-1).imag[:,:,0:self.input_dim//2])+1e-8)  , torch.log(torch.abs(torch.fft.fft(x , dim=-1).real[:,:,0:self.input_dim//2])+1e-8) ] , dim=-1)

        x = x.transpose(1,2)

        B, L, C = x.shape

        

        dec_inp = self.dec_inp[:B]

        x = self.input_layer(x)
        embed_shape = x.shape

        x = self.positional_encoding(x)


        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # x = x.transpose(0,1)

        # x = self.mlp_head(x)

        dec_in = self.input_layer_dec(dec_inp)
        dec_in = self.positional_encoding_dec(dec_in)

        dec_in = dec_in.transpose(0,1)

        dec_out = self.transformerdecoder(x, dec_in)
        x = dec_out.transpose(0,1)

        

        

        # Perform classification prediction
        # cls = x[0]
        # out = self.mlp_head(x.reshape(B,-1))
        return x

    # def forward_feature(self, x):
    #     # Preprocess input
        
    #     # x = torch.log( torch.abs( torch.fft.fft(x , dim=-1) ) + 1e-10 )
    #     # print(x.shape)
    #     # print("check2:",torch.isneginf(x).any(),x.dtype)

    #     xshape = x.shape

    #     x_flat = x.view(-1, xshape[1])  

    #     output_flat = self.init_norm(x_flat)

    #     x = output_flat.reshape(xshape)

    #     # print(x.mean(dim=1 , keepdim=True))

    #     # # x = self.init_norm(x)
        
    #     # fft = torch.fft.fft(x , dim=-1)
    #     # # print(x)
    #     # # print(fft)

    #     # x = torch.cat( [torch.log(torch.abs( torch.fft.fft(x , dim=-1).imag[:,:,0:self.input_dim//2])+1e-8)  , torch.log(torch.abs(torch.fft.fft(x , dim=-1).real[:,:,0:self.input_dim//2])+1e-8) ] , dim=-1)
    #     # print("check3:",torch.isneginf(x).any())
    #     # x = (x - x.mean(dim=1 , keepdim=True) )/x.std(dim=1 , keepdim=True) 
    #     # print(x)
    #     B, T, D = x.shape
    #     x = self.input_layer(x)
    #     embed_shape = x.shape

        

    #     # Add CLS token and positional encoding
    #     cls_token = self.cls_token.repeat(B, 1, 1)
    #     # print(x.shape)
    #     x_flat = x.reshape((x.shape[0],x.shape[-1]*self.cut,-1))

    #     for i in range(x_flat.shape[-1]):
    #         channel_token_emb_i = (
    #                 self.channel_tokens(self.index[i])
    #                 .unsqueeze(0)
    #                 .repeat(B, self.cut)
    #             )
    #         x_flat[:,:,i] =  x_flat[:,:,i] + channel_token_emb_i

    #     x_flat = self.positional_encoding(x_flat)

    #     # x_flat = x_flat + self.pos_embedding
    #     # x_flat = x.reshape((x.shape[0],-1,x.shape[-1]*self.cut))
    #     # x_flat = x_flat + self.channel_embedding

    #     # print(x_flat.shape)

    #     x = x_flat.reshape(embed_shape)

    #     # x = torch.cat([cls_token, x], dim=1)
    #     # x = x + self.pos_embedding[:, : T + 1]

    #     # Apply Transforrmer
    #     x = self.dropout(x)
    #     x = x.transpose(0, 1)
    #     x = self.transformer(x)

    #     x = x.transpose(0,1).mean(dim=1)

        

    #     # Perform classification prediction
    #     # cls = x[0]
    #     # out = self.mlp_head(x.reshape(B,-1))
    #     return x