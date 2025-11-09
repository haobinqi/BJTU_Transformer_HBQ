# src/data.py
import os
import re
import sentencepiece as spm
from xml.etree import ElementTree as ET
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader

def extract_segments_from_xml(path, lang="en"):
    """
    解析 IWSLT-style xml 文件，返回每个 <seg> 内容（或 train 文件没有 seg 标签则按行返回）
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 如果文件中含有 <seg id=...> 标签，使用 ElementTree 解析
    if '<seg' in text:
        root = ET.fromstring(text)
        segs = []
        for seg in root.iter('seg'):
            segs.append(seg.text.strip())
        return segs
    else:
        # 非xml，按行划分（train 的长 doc 文本）
        lines = []
        for line in text.splitlines():
            l = line.strip()
            if l:
                lines.append(l)
        return lines

def train_sentencepiece(corpus_files, model_prefix="spm", vocab_size=8000, model_type='unigram'):
    # 合并文件到临时文件
    tmp = model_prefix + "_corpus.txt"
    with open(tmp, 'w', encoding='utf-8') as fout:
        for f in corpus_files:
            with open(f, 'r', encoding='utf-8') as fin:
                for l in fin:
                    l = l.strip()
                    if l:
                        fout.write(l + '\n')
    spm.SentencePieceTrainer.Train(f'--input={tmp} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type} --character_coverage=1.0')
    os.remove(tmp)
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, sp_src, sp_tgt, max_len=100):
        assert len(src_lines) == len(tgt_lines)
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len
        # ids for special tokens
        self.bos = sp_tgt.PieceToId('<s>') if sp_tgt.PieceToId('<s>') >= 0 else 1
        self.eos = sp_tgt.PieceToId('</s>') if sp_tgt.PieceToId('</s>') >= 0 else 2

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.sp_src.encode(self.src_lines[idx], out_type=int)
        tgt = self.sp_tgt.encode(self.tgt_lines[idx], out_type=int)
        if len(src) > self.max_len: src = src[:self.max_len]
        if len(tgt) > self.max_len-2: tgt = tgt[:self.max_len-2]
        # add bos/eos
        tgt = [self.bos] + tgt + [self.eos]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch, pad_id_src=0, pad_id_tgt=0):
    srcs, tgts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)
    src_batch = torch.full((len(batch), max_src), pad_id_src, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt), pad_id_tgt, dtype=torch.long)
    for i, (s,t) in enumerate(zip(srcs,tgts)):
        src_batch[i, :len(s)] = s
        tgt_batch[i, :len(t)] = t
    return src_batch, tgt_batch

def make_dataloader(src_lines, tgt_lines, sp_src, sp_tgt, batch_size=32, shuffle=True, max_len=100):
    dataset = TranslationDataset(src_lines, tgt_lines, sp_src, sp_tgt, max_len=max_len)
    pad_src = 0
    pad_tgt = 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: collate_fn(b, pad_src, pad_tgt))
