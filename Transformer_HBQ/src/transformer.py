# src/transformer.py
# 训练辅助：mask, label smoothing, NoamOpt, greedy_decode, decode 文本清洗
import torch
import torch.nn as nn
import math

def subsequent_mask(size):
    """
    返回 (1, size, size) 的布尔矩阵，用于 decoder 阶段的未来位置屏蔽
    True 表示可以 attend，False 表示 mask 掉
    """
    attn_shape = (1, size, size)
    # 上三角（不含主对角）为 1 表示禁止
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~subsequent  # True where allowed

def create_masks(src, tgt, pad_idx=0, device='cpu'):
    """
    src: (B, src_len)
    tgt: (B, tgt_len)
    返回:
      src_mask: (B, 1, src_len) 布尔，True 表示非 pad（可 attend）
      tgt_mask: (B, tgt_len, tgt_len) 布尔，True 表示可 attend
    """
    src_mask = (src != pad_idx).unsqueeze(1).to(device)  # (B,1,src_len)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).to(device)  # (B,1,tgt_len)
    size = tgt.size(1)
    subseq = subsequent_mask(size).to(device)  # (1, size, size)
    # tgt_mask: (B, tgt_len, tgt_len) = tgt_pad_mask (B,1,tgt_len) & subseq (1,tgt_len,tgt_len)
    tgt_mask = tgt_pad_mask & subseq
    return src_mask, tgt_mask

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab, ignore_index=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.smoothing = label_smoothing
        self.tgt_vocab = tgt_vocab
        self.ignore_index = ignore_index

    def forward(self, pred_logprob, target):
        # pred_logprob: log softmax (batch*seq, vocab)
        confidence = 1.0 - self.smoothing
        true_dist = pred_logprob.data.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab - 1))
        mask = (target != self.ignore_index)
        idx = target.unsqueeze(1)
        true_dist.scatter_(1, idx, confidence)
        true_dist[~mask] = 0
        return self.criterion(pred_logprob, true_dist)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_symbol=None, device='cpu'):
    """
    批量贪心解码：
    - src: (B, src_len)
    - src_mask: (B, 1, src_len) boolean
    返回: (B, T) ids（包含 start_symbol 开头，可能包含 eos）
    会在每个样本生成 eos 时为该样本停止生成（其他样本继续），当所有样本都完成时提前结束
    """
    model.eval()
    B = src.size(0)
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.full((B, 1), start_symbol, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, memory, src_mask, tgt_mask)  # (B, seq, d_model)
            logits = model.out(out[:, -1, :])  # (B, vocab)
            next_word = torch.argmax(logits, dim=-1).unsqueeze(1)  # (B,1)
            ys = torch.cat([ys, next_word], dim=1)
            if eos_symbol is not None:
                finished = finished | (next_word.squeeze(1) == eos_symbol)
                if finished.all():
                    break
    return ys  # (B, seq_len)

def clean_decode_ids(sp_model, ids, pad_id=0, eos_id=None):
    """
    将一维 id 列表转换为可读字符串（删除 pad、截断 eos）
    ids: list[int] 或 tensor
    返回 decode 后的字符串
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    # remove leading pads (unlikely) and keep until eos
    out = []
    for iid in ids:
        if iid == pad_id:
            continue
        out.append(iid)
        if eos_id is not None and iid == eos_id:
            break
    # 防护：去掉非法 id（负数）
    out = [i for i in out if isinstance(i, int) and i >= 0]
    try:
        return sp_model.decode(out).strip()
    except Exception:
        # fallback: join numeric ids as string
        return " ".join(map(str, out))
