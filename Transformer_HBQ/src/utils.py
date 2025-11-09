# src/utils.py
# 小修正：保留原有 API，同时添加一个属性风格的 avg_value 以便不同调用习惯
import os
import random
import math
import time
import json
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, device):
    return torch.load(path, map_location=device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter:
    """
    记录累积值与计数
    update(val, n=1) -> 增加 sum += val * n, count += n
    avg() -> 返回平均 (兼容历史调用)
    value (property) -> 返回平均（属性风格）
    """
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        # val 是每个样本的平均 loss（或 metric），n 是样本数
        self.sum += float(val) * int(n)
        self.count += int(n)

    def avg(self):
        # 兼容你现有代码使用 epoch_loss_meter.avg()
        return self.sum / max(1, self.count)

    @property
    def value(self):
        return self.avg()
