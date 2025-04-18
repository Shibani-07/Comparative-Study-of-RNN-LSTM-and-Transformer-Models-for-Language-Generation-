# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embed, seq_len_max=512):
        super().__init__()
        position = torch.arange(seq_len_max).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embed, 2) * (-math.log(10000.0) / dim_embed))
        pe = torch.zeros(1, seq_len_max, dim_embed)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_hidden, num_blocks, n_heads, seq_len_max=512, p_drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embed)
        self.pos_encoder = PositionalEncoding(dim_embed, seq_len_max)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embed,
            nhead=n_heads,
            dim_feedforward=dim_hidden,
            dropout=p_drop,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_blocks)
        self.output = nn.Linear(dim_embed, vocab_size)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        return self.output(x)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_hidden, num_blocks, p_drop=0.1, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embed)
        self.lstm = nn.LSTM(
            input_size=dim_embed,
            hidden_size=dim_hidden,
            num_layers=num_blocks,
            dropout=p_drop if num_blocks > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(p_drop)
        self.output = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, attention_mask=None):
        x = self.dropout(self.embedding(x))
        x, _ = self.lstm(x)
        return self.output(x)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_hidden, num_blocks, p_drop=0.1, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embed)
        self.rnn = nn.RNN(
            input_size=dim_embed,
            hidden_size=dim_hidden,
            num_layers=num_blocks,
            dropout=p_drop if num_blocks > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(p_drop)
        self.output = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, attention_mask=None):
        x = self.dropout(self.embedding(x))
        x, _ = self.rnn(x)
        return self.output(x)
