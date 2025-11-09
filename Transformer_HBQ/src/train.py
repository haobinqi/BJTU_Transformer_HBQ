# ============================================================
# train.py (支持完整消融实验版本)
# ============================================================

import os
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import data as data_module
import model as model_module
import transformer as transformer_module
from utils import set_seed, save_checkpoint, load_checkpoint, count_parameters, AverageMeter

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

plt.switch_backend('Agg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/iwslt2017/', help="训练/验证数据路径")
    parser.add_argument('--outdir', type=str, default='./results', help="结果输出目录")
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--vocab_size', type=int, default=8000)

    # 架构消融实验参数
    parser.add_argument('--no_positional_encoding', action='store_true')
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--no_layernorm', action='store_true')
    parser.add_argument('--single_head', action='store_true')
    parser.add_argument('--encoder_only', action='store_true', help="仅使用Encoder (BERT风格)")
    parser.add_argument('--decoder_only', action='store_true', help="仅使用Decoder (GPT风格)")

    # 训练技巧消融实验参数
    parser.add_argument('--no_grad_clip', action='store_true', help="无梯度裁剪")
    parser.add_argument('--no_lr_schedule', action='store_true', help="无学习率调度")
    parser.add_argument('--no_weight_decay', action='store_true', help="无权重衰减")
    parser.add_argument('--fixed_lr', type=float, default=None, help="固定学习率")

    args = parser.parse_args()
    return args


# --------------------------
# 加载数据 & SentencePiece
# --------------------------
def load_data_and_spm(data_dir, spm_prefix_src='spm_src', spm_prefix_tgt='spm_tgt', vocab_size=8000):
    src_file = os.path.join(data_dir, 'train.en')
    tgt_file = os.path.join(data_dir, 'train.de')
    src_lines = data_module.extract_segments_from_xml(src_file, lang='en')
    tgt_lines = data_module.extract_segments_from_xml(tgt_file, lang='de')

    val_src_file = os.path.join(data_dir, 'val.en')
    val_tgt_file = os.path.join(data_dir, 'val.de')
    val_src_lines, val_tgt_lines = None, None
    if os.path.exists(val_src_file) and os.path.exists(val_tgt_file):
        val_src_lines = data_module.extract_segments_from_xml(val_src_file, lang='en')
        val_tgt_lines = data_module.extract_segments_from_xml(val_tgt_file, lang='de')

    sp_src = data_module.train_sentencepiece([src_file], model_prefix=spm_prefix_src, vocab_size=vocab_size)
    sp_tgt = data_module.train_sentencepiece([tgt_file], model_prefix=spm_prefix_tgt, vocab_size=vocab_size)
    return src_lines, tgt_lines, val_src_lines, val_tgt_lines, sp_src, sp_tgt


# --------------------------
# 仅Encoder模型 (BERT风格)
# --------------------------
class EncoderOnlyModel(nn.Module):
    """仅使用Encoder的模型，用于分类或序列标注任务"""

    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.pos_encoding = model_module.PositionalEncoding(d_model)

        attn = model_module.MultiHeadedAttention(h, d_model, dropout)
        ff = model_module.PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder = model_module.Encoder(
            model_module.EncoderLayer(d_model, attn, ff, dropout), N
        )
        # 输出层：使用[CLS] token或平均池化
        self.classifier = nn.Linear(d_model, tgt_vocab)

        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        x = self.src_embed(src) * math.sqrt(self.src_embed.embedding_dim)
        x = self.pos_encoding(x)
        encoded = self.encoder(x, src_mask)

        # 使用第一个token ([CLS]) 作为分类特征
        cls_output = encoded[:, 0, :]
        return self.classifier(cls_output)


# --------------------------
# 仅Decoder模型 (GPT风格)
# --------------------------
class DecoderOnlyModel(nn.Module):
    """仅使用Decoder的模型，用于语言建模任务"""

    def __init__(self, tgt_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_encoding = model_module.PositionalEncoding(d_model)

        attn = model_module.MultiHeadedAttention(h, d_model, dropout)
        ff = model_module.PositionwiseFeedForward(d_model, d_ff, dropout)

        self.decoder = model_module.Decoder(
            model_module.DecoderLayer(d_model, attn, attn, ff, dropout), N
        )
        self.out = nn.Linear(d_model, tgt_vocab)

        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, tgt_mask):
        x = self.tgt_embed(tgt) * math.sqrt(self.tgt_embed.embedding_dim)
        x = self.pos_encoding(x)
        decoded = self.decoder(x, None, None, tgt_mask)
        return self.out(decoded)


# --------------------------
# Greedy 解码
# --------------------------
def greedy_decode_batch(model, src_batch, src_mask, max_len, bos_id, eos_id, device, encoder_only=False,
                        decoder_only=False):
    model.eval()
    B = src_batch.size(0)

    with torch.no_grad():
        if encoder_only:
            # Encoder-only模型使用分类方式
            logits = model(src_batch, src_mask)
            predictions = torch.argmax(logits, dim=-1)
            return predictions.unsqueeze(1)  # 简单返回分类结果

        elif decoder_only:
            # Decoder-only模型自回归生成
            ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                tgt_mask = transformer_module.subsequent_mask(ys.size(1)).to(device)
                out = model(ys, tgt_mask)
                probs = out[:, -1, :]
                next_word = torch.argmax(probs, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if eos_id is not None:
                    finished = finished | (next_word.squeeze(1) == eos_id)
                    if finished.all():
                        break
            return ys

        else:
            # 完整Encoder-Decoder模型
            memory = model.encode(src_batch, src_mask)
            ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                tgt_mask = transformer_module.subsequent_mask(ys.size(1)).to(device)
                out = model.decode(ys, memory, src_mask, tgt_mask)
                probs = model.out(out[:, -1, :])
                next_word = torch.argmax(probs, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if eos_id is not None:
                    finished = finished | (next_word.squeeze(1) == eos_id)
                    if finished.all():
                        break
            return ys


# --------------------------
# 评估函数
# --------------------------
def evaluate(model, dataloader, criterion, device, sp_tgt=None, max_len=80, pad_idx=0, bos_idx=1, eos_idx=2,
             encoder_only=False, decoder_only=False):
    model.eval()
    total_loss = 0.0
    references, hypotheses = [], []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            if encoder_only:
                # Encoder-only: 分类任务
                src_mask = (src_batch != pad_idx).unsqueeze(1).to(device)
                out_logits = model(src_batch, src_mask)
                # 简化处理：假设每个序列对应一个标签
                loss = criterion(out_logits, tgt_batch[:, 0])  # 使用第一个token作为标签

                # 生成预测
                pred_ids = torch.argmax(out_logits, dim=-1).unsqueeze(1)

            elif decoder_only:
                # Decoder-only: 语言建模
                tgt_input = tgt_batch[:, :-1]
                tgt_gold = tgt_batch[:, 1:]
                tgt_mask = transformer_module.subsequent_mask(tgt_input.size(1)).to(device)
                out_logits = model(tgt_input, tgt_mask)
                loss = criterion(out_logits.reshape(-1, out_logits.size(-1)), tgt_gold.reshape(-1))

                # 生成预测
                src_mask = (src_batch != pad_idx).unsqueeze(1).to(device)
                pred_ids = greedy_decode_batch(model, src_batch, src_mask, max_len, bos_idx, eos_idx, device,
                                               decoder_only=True)
            else:
                # 完整Encoder-Decoder
                tgt_input = tgt_batch[:, :-1]
                tgt_gold = tgt_batch[:, 1:]
                src_mask, tgt_mask = transformer_module.create_masks(src_batch, tgt_input, pad_idx=pad_idx,
                                                                     device=device)
                out_logits = model(src_batch, tgt_input, src_mask, tgt_mask)
                loss = criterion(out_logits.reshape(-1, out_logits.size(-1)), tgt_gold.reshape(-1))

                # 生成预测
                pred_ids = greedy_decode_batch(model, src_batch, src_mask, max_len, bos_idx, eos_idx, device)

            total_loss += loss.item() * src_batch.size(0)

            # 转换为文本
            for ref_ids, hyp_ids in zip(tgt_batch.cpu().tolist(), pred_ids.cpu().tolist()):
                ref_txt = sp_tgt.decode([i for i in ref_ids if i != pad_idx]).strip()
                if encoder_only:
                    # Encoder-only: 只处理单个token预测
                    hyp_txt = sp_tgt.decode([hyp_ids[0] if hyp_ids[0] != pad_idx else '']).strip()
                else:
                    hyp_txt = sp_tgt.decode([i for i in hyp_ids if i != pad_idx]).strip()

                references.append([ref_txt.split()])
                hypotheses.append(hyp_txt.split())

    avg_loss = total_loss / len(dataloader.dataset)
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn) if len(hypotheses) > 0 else 0.0
    return avg_loss, bleu


# --------------------------
# 绘图保存
# --------------------------
def plot_curves(train_losses, val_losses, bleu_scores, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch");
    plt.ylabel("Loss");
    plt.title("Train & Val Loss")
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    if bleu_scores:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, label="Val BLEU", marker='s')
        plt.xlabel("Epoch");
        plt.ylabel("BLEU");
        plt.title("Validation BLEU")
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bleu_score.png"))
        plt.close()
    print(f"Saved plots to {save_dir}")


# --------------------------
# 训练主函数
# --------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)

    src_lines, tgt_lines, val_src_lines, val_tgt_lines, sp_src, sp_tgt = load_data_and_spm(
        args.data_dir, vocab_size=args.vocab_size)

    train_loader = data_module.make_dataloader(src_lines, tgt_lines, sp_src, sp_tgt,
                                               batch_size=args.batch_size, shuffle=True, max_len=80)
    val_loader = None
    if val_src_lines and val_tgt_lines:
        val_loader = data_module.make_dataloader(val_src_lines, val_tgt_lines, sp_src, sp_tgt,
                                                 batch_size=args.batch_size, shuffle=False, max_len=80)

    device = torch.device(args.device)

    # 模型选择
    if args.encoder_only:
        print("🚀 Using Encoder-Only Model (BERT-style)")
        model = EncoderOnlyModel(
            src_vocab=sp_src.get_piece_size(),
            tgt_vocab=sp_tgt.get_piece_size(),
            d_model=args.d_model,
            N=args.num_layers,
            h=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)
    elif args.decoder_only:
        print("🚀 Using Decoder-Only Model (GPT-style)")
        model = DecoderOnlyModel(
            tgt_vocab=sp_tgt.get_piece_size(),
            d_model=args.d_model,
            N=args.num_layers,
            h=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)
    else:
        print("🚀 Using Full Encoder-Decoder Model")
        model = model_module.Transformer(
            src_vocab=sp_src.get_piece_size(),
            tgt_vocab=sp_tgt.get_piece_size(),
            d_model=args.d_model,
            N=args.num_layers,
            h=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            use_positional_encoding=not args.no_positional_encoding,
            use_residual=not args.no_residual,
            use_layernorm=not args.no_layernorm,
            single_head=args.single_head
        ).to(device)

    print("Parameters:", count_parameters(model))
    pad_id = sp_tgt.piece_to_id("<pad>") if hasattr(sp_tgt, "piece_to_id") else 0
    bos_id = sp_tgt.piece_to_id("<s>") if hasattr(sp_tgt, "piece_to_id") else 1
    eos_id = sp_tgt.piece_to_id("</s>") if hasattr(sp_tgt, "piece_to_id") else 2

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 优化器配置
    weight_decay = 0.0 if args.no_weight_decay else 1e-2
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # 学习率调度配置
    if args.no_lr_schedule or args.fixed_lr is not None:
        print("📊 Using Fixed Learning Rate")
        if args.fixed_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.fixed_lr
        scheduler = None
    else:
        print("📊 Using Noam Learning Rate Schedule")
        scheduler = transformer_module.NoamOpt(args.d_model, factor=1, warmup=4000, optimizer=optimizer)

    global_step = 0
    train_losses, val_losses, val_bleus = [], [], []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_meter = AverageMeter()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for src_batch, tgt_batch in loop:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            if args.encoder_only:
                # Encoder-only训练
                src_mask = (src_batch != pad_id).unsqueeze(1).to(device)
                out_logits = model(src_batch, src_mask)
                # 简化：使用第一个token作为标签
                loss = criterion(out_logits, tgt_batch[:, 0])

            elif args.decoder_only:
                # Decoder-only训练
                tgt_input = tgt_batch[:, :-1]
                tgt_gold = tgt_batch[:, 1:]
                tgt_mask = transformer_module.subsequent_mask(tgt_input.size(1)).to(device)
                out_logits = model(tgt_input, tgt_mask)
                loss = criterion(out_logits.reshape(-1, out_logits.size(-1)), tgt_gold.reshape(-1))

            else:
                # 完整Encoder-Decoder训练
                tgt_input, tgt_gold = tgt_batch[:, :-1], tgt_batch[:, 1:]
                src_mask, tgt_mask = transformer_module.create_masks(src_batch, tgt_input, pad_idx=pad_id,
                                                                     device=device)
                out_logits = model(src_batch, tgt_input, src_mask, tgt_mask)
                loss = criterion(out_logits.reshape(-1, out_logits.size(-1)), tgt_gold.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if not args.no_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()
            else:
                optimizer.step()

            epoch_loss_meter.update(loss.item(), n=src_batch.size(0))
            global_step += 1

        train_losses.append(epoch_loss_meter.avg())
        print(f"Epoch {epoch + 1} training loss: {epoch_loss_meter.avg():.4f}")

        # -------- 每 5 轮保存 checkpoint --------
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.outdir, "checkpoints", f"ckpt_epoch{epoch + 1}.pt")
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch + 1}, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        # 验证集
        if val_loader:
            val_loss, val_bleu = evaluate(model, val_loader, criterion, device, sp_tgt=sp_tgt,
                                          max_len=80, pad_idx=pad_id, bos_idx=bos_id, eos_idx=eos_id,
                                          encoder_only=args.encoder_only, decoder_only=args.decoder_only)
            val_losses.append(val_loss)
            val_bleus.append(val_bleu)
            print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}, BLEU: {val_bleu:.4f}")

    # -------- 训练结束后保存最终模型到 results 目录 --------
    final_model_path = os.path.join(args.outdir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Saved final model to {final_model_path}")

    plot_curves(train_losses, val_losses, val_bleus, save_dir=args.outdir)
    print("Training finished.")


if __name__ == "__main__":
    args = parse_args()
    train(args)

