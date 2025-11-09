# src/model.py
# 手工实现 Transformer（Encoder + Decoder）
# 主要修正：
#  - ScaledDotProductAttention 内部构建 Dropout 模块并使用
#  - MultiHeadedAttention 在 __init__ 创建 attn_layer 避免取 self.dropout.p 的错误
#  - 支持消融实验：无位置编码、无残差连接、无层归一化、单头注意力

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class IdentityLayerNorm(nn.Module):
    """用于消融实验的恒等层归一化"""

    def __init__(self, features, eps=1e-6):
        super().__init__()

    def forward(self, x):
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, use_residual=True, use_layernorm=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = LayerNorm(size)
        else:
            self.norm = IdentityLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        if self.use_residual:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention，实现内部分布式 dropout（传入 dropout 概率）
    query/key/value shapes: (batch, heads, seq_len, d_k)
    mask: boolean mask broadcastable to (batch, heads, seq_len, seq_len)
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, heads, seq, seq)
        if mask is not None:
            # mask should be boolean: True = keep, False = mask out
            scores = scores.masked_fill(~mask, float('-1e9'))
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, single_head=False):
        super().__init__()
        if single_head:
            h = 1  # 单头注意力消融实验
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # Q, K, V, and final linear
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn_layer = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # linear projections and split heads
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]
        if mask is not None:
            # mask: (batch, 1, seq_len) or (batch, seq_len, seq_len) -> expand to (batch, heads, seq_len, seq_len)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch,1,seq_len,seq_len) after broadcast
            else:
                mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.h, 1, 1)
        x, self.attn = self.attn_layer(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, use_residual=True, use_layernorm=True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, use_residual, use_layernorm), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, use_residual=True, use_layernorm=True):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, use_residual, use_layernorm), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N, use_layernorm=True):
        super().__init__()
        self.layers = clones(layer, N)
        if use_layernorm:
            self.norm = LayerNorm(layer.size)
        else:
            self.norm = IdentityLayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N, use_layernorm=True):
        super().__init__()
        self.layers = clones(layer, N)
        if use_layernorm:
            self.norm = LayerNorm(layer.size)
        else:
            self.norm = IdentityLayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1,
                 use_positional_encoding=True, use_residual=True, use_layernorm=True, single_head=False):
        super().__init__()

        # 词嵌入层
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)

        # 位置编码（可选）
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model)
        else:
            self.pos_encoding = nn.Identity()  # 无位置编码消融实验

        # 注意力机制（支持单头消融实验）
        attn = MultiHeadedAttention(h, d_model, dropout, single_head=single_head)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 编码器和解码器
        self.encoder = Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout,
                                            use_residual, use_layernorm), N, use_layernorm)
        self.decoder = Decoder(
            DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout,
                         use_residual, use_layernorm), N, use_layernorm)
        self.out = nn.Linear(d_model, tgt_vocab)

        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim)
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim)
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        dec = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.out(dec)