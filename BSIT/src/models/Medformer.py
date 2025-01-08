import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Medformer_EncDec import Encoder, EncoderLayer, MedformerLayer, ListPatchEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = True
        self.single_channel = False
        self.n_channels = configs.num_patches // configs.num_cut
        self.enc_in = self.n_channels
        self.cut = configs.num_cut
        # Embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.num_t_pints * configs.num_cut
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            self.n_channels,
            configs.embed_dim,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            self.single_channel,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        configs.embed_dim,
                        configs.num_heads,
                        configs.dropout,
                        self.output_attention,
                        False,
                    ),
                    configs.embed_dim,
                    configs.hidden_dim,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.embed_dim),
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.embed_dim
            * sum(patch_num_list)
            * (1 if not self.single_channel else configs.n_channels),
            configs.n_classes,
        )

    def forward(self, x_enc, train = False):
        # Embedding
        B,T,D = x_enc.shape
        x_enc = x_enc.reshape((B,-1 , self.n_channels))

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        embed = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(embed)  # (batch_size, num_classes)
        return output, embed.cpu().detach().numpy()
