# ============================================================
# eval.py — 与 train 验证逻辑一致，输出 BLEU 与平均 loss
# ============================================================

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sacrebleu import corpus_bleu as sacrebleu_corpus

import data as data_module
import model as model_module
import transformer as transformer_module
from utils import load_checkpoint


# -------------------------------
# Greedy Decode（同 train.py）
# -------------------------------
def greedy_decode_batch(model, src_batch, src_mask, max_len, bos_id, eos_id, device):
    model.eval()
    B = src_batch.size(0)
    with torch.no_grad():
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


# -------------------------------
# Evaluation（计算 loss + BLEU）
# -------------------------------
def evaluate(model, dataloader, device, sp_tgt, max_len=80, pad_idx=0, bos_idx=1, eos_idx=2):
    model.eval()
    total_loss, total_tokens = 0, 0
    references, hypotheses = [], []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dataloader, desc="Evaluating"):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            src_mask = (src_batch != pad_idx).unsqueeze(1)

            # ====== 1️⃣ 计算 loss（与 train 一致） ======
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]
            tgt_mask = transformer_module.subsequent_mask(tgt_input.size(1)).to(device)

            logits = model(src_batch, tgt_input, src_mask, tgt_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=pad_idx,
                reduction='sum'
            )

            num_tokens = (tgt_output != pad_idx).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens

            # ====== 2️⃣ 解码 & BLEU ======
            pred_ids = greedy_decode_batch(model, src_batch, src_mask, max_len, bos_idx, eos_idx, device)
            for ref_ids, hyp_ids in zip(tgt_batch.cpu().tolist(), pred_ids.cpu().tolist()):
                ref_txt = sp_tgt.decode([i for i in ref_ids if i != pad_idx]).strip()
                hyp_txt = sp_tgt.decode([i for i in hyp_ids if i != pad_idx]).strip()
                references.append([ref_txt.split()])
                hypotheses.append(hyp_txt.split())

    avg_loss = total_loss / total_tokens
    nltk_bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)
    refs_flat = [" ".join(r[0]) for r in references]
    hyps_flat = [" ".join(h) for h in hypotheses]
    sacre_bleu = sacrebleu_corpus(hyps_flat, [refs_flat]).score

    print(f"\nTest Loss (average per token): {avg_loss:.4f}")
    print(f"Validation-style BLEU (NLTK) : {nltk_bleu:.4f}")
    print(f"SacreBLEU (standard)         : {sacre_bleu:.4f}")

    return avg_loss, nltk_bleu, sacre_bleu, hypotheses


# -------------------------------
# 主函数
# -------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径配置
    test_src_file = './dataset/iwslt2017/test.en'
    test_tgt_file = './dataset/iwslt2017/test.de'
    checkpoint_path = './results/checkpoints/ckpt_epoch30.pt'
    spm_src_path = './results/spm_src.model'
    spm_tgt_path = './results/spm_tgt.model'
    batch_size = 16
    max_len = 80

    import sentencepiece as spm
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(spm_src_path)
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(spm_tgt_path)

    # 模型初始化（保持与训练一致）
    model = model_module.Transformer(
        src_vocab=sp_src.get_piece_size(),
        tgt_vocab=sp_tgt.get_piece_size(),
        d_model=256,
        N=4,
        h=8,
        d_ff=1024,
        dropout=0.1
    ).to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"✅ Loaded model from {checkpoint_path}")

    # 数据准备
    src_lines = [line.strip() for line in open(test_src_file, encoding='utf-8')]
    tgt_lines = [line.strip() for line in open(test_tgt_file, encoding='utf-8')]

    test_loader = data_module.make_dataloader(
        src_lines, tgt_lines, sp_src, sp_tgt,
        batch_size=batch_size, shuffle=False, max_len=max_len
    )

    print(f"Prepared test loader with {len(test_loader.dataset)} examples, batch_size={batch_size}")

    # token ids
    pad_id = sp_tgt.piece_to_id("<pad>") if sp_tgt.piece_to_id("<pad>") >= 0 else 0
    bos_id = sp_tgt.piece_to_id("<s>") if sp_tgt.piece_to_id("<s>") >= 0 else 1
    eos_id = sp_tgt.piece_to_id("</s>") if sp_tgt.piece_to_id("</s>") >= 0 else 2

    # 评估
    avg_loss, nltk_bleu, sacre_bleu, hypotheses = evaluate(
        model, test_loader, device, sp_tgt,
        max_len=max_len, pad_idx=pad_id, bos_idx=bos_id, eos_idx=eos_id
    )

    # 保存预测
    os.makedirs('./results', exist_ok=True)
    save_path = './results/test_predictions.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        for h in hypotheses:
            f.write(" ".join(h) + "\n")

    print(f"\nSaved predictions to {save_path}")
    print(f"✅ Final Test Loss  : {avg_loss:.4f}")
    print(f"✅ Final Test BLEU  : {nltk_bleu:.4f}")
    print(f"✅ Final SacreBLEU  : {sacre_bleu:.4f}")


if __name__ == "__main__":
    main()
