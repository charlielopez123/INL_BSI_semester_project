
import torch.nn as nn
import torch
from fast_transformers.attention.attention_layer import AttentionLayer
from models.linear_att_lib import LinearAttention
from fast_transformers.masking import FullMask
from fast_transformers.masking import LengthMask

class AttentionBlock(nn.Module):
    def __init__(self, args, embed_dim, num_patches, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.linearattn = LinearAttention(embed_dim)
        self.mask = FullMask(num_patches)
        query_lengths = torch.full((args.batch_size,), num_patches, dtype=torch.long)  # 表示第一个序列长度为8，第二个为5
        key_lengths = query_lengths.clone()  # 假设 key 和 query 的长度相同

        self.query_mask = LengthMask(query_lengths)
        self.key_mask = LengthMask(key_lengths)

        self.attn = AttentionLayer(self.linearattn, embed_dim, num_heads)
        # self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        inp_x = x
        x = self.layer_norm_1(x + self.dropout(self.attn(inp_x, inp_x, inp_x,self.mask,self.query_mask,self.key_mask)))
        x = self.layer_norm_2(x + self.dropout(self.linear(x)))
        return x


class AttentionDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_patches, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.layer_norm_3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, dec_inp):
        
        dec_inp = self.layer_norm_1(dec_inp + self.dropout(self.self_attn(dec_inp, dec_inp, dec_inp)[0]))

        # inp_x = self.layer_norm_1(x)
        # print(dec_inp.shape,x.shape)
        dec_out = self.layer_norm_2(dec_inp + self.dropout(self.cross_attn(dec_inp, x, x)[0]))
        dec_out = self.layer_norm_3(dec_out + self.dropout(self.linear(dec_out)))
        return dec_out