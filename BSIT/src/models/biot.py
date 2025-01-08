import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.attention import AttentionBlock


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
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
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        emb_size = args.embed_dim
        heads = args.num_heads
        self.task = args.task
        self.cut = args.num_cut
        depth = args.num_layers
        n_channels = args.num_patches // args.num_cut
        hidden_dim = args.hidden_dim
        n_fft=8
        hop_length=4
        dropout = args.dropout
        num_classes = args.n_classes

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = nn.Sequential(
            *(AttentionBlock(emb_size, args.num_patches, hidden_dim, heads, dropout=dropout) for _ in range(depth))
        )

        self.classifier = ClassificationHead(emb_size, num_classes)

        self.embedding_dim = args.embed_dim
        self.embedding_predict_dim = self.embedding_dim // 2

        self.predictor = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_predict_dim, bias=False),
                                nn.BatchNorm1d(self.embedding_predict_dim),
                                nn.ReLU(inplace=True), # hidden layer
                                nn.Linear(self.embedding_predict_dim, self.embedding_dim)) # output layer
        
        # self.transformer = LinearAttentionTransformer(
        #     dim=emb_size,
        #     heads=heads,
        #     depth=depth,
        #     max_seq_len=1024,
        #     attn_layer_dropout=0.2,  # dropout right after self-attention layer
        #     attn_dropout=0.2,  # dropout post-attention
        # )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        x = x.reshape((x.shape[0],-1,x.shape[-1]*self.cut))
        x = (x - x.mean(dim=1 , keepdim=True) )/x.std(dim=1 , keepdim=True)
        
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            # print(i)
            # print(self.index[i])
            channel_token_emb = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            # if perturb:
            #     ts = channel_emb.shape[1]
            #     ts_new = np.random.randint(ts // 2, ts)
            #     selected_ts = np.random.choice(range(ts), ts_new, replace=False)
            #     channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = emb.transpose(0, 1)

        emb = self.transformer(emb)
        emb = emb.transpose(0,1).mean(dim=1)
        logit = self.classifier(emb)
        if self.task != "SSLEval":
            emb = emb.cpu().detach().numpy()
        return logit,emb
