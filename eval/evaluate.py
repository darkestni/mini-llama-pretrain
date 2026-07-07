"""
evaluate.py
===========
预训练模型评测：计算困惑度(perplexity) + 文本生成对比（随机初始 vs 训后）。

困惑度是预训练的核心指标——衡量模型对语言的"困惑程度"。
    - 随机初始：PPL ~30000（完全不懂语言）
    - 训得好的：PPL 降到几十（能流利生成）

用法：
    # 评测训后模型
    python eval/evaluate.py --model outputs/pretrain_run

    # 对比随机初始 baseline
    python eval/evaluate.py --baseline

    # Mac 离线走通
    python eval/evaluate.py --offline
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from config import build_model, count_parameters  # noqa: E402


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================
# 困惑度计算
# ============================================================
def compute_perplexity(model, dataset, tokenizer, batch_size=4, device="cuda"):
    """
    计算模型在数据集上的困惑度。
    PPL = exp(avg_loss)
    """
    import torch
    model.eval()
    total_loss = 0
    total_tokens = 0
    n = len(dataset)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = dataset[i:i+batch_size]
            input_ids = torch.tensor(batch, dtype=torch.long).to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            # loss 是 per-token 平均
            seq_len = input_ids.numel()
            total_loss += outputs.loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    return {"loss": avg_loss, "perplexity": ppl}


# ============================================================
# 文本生成对比
# ============================================================
PROMPTS = [
    "机器学习",
    "人工智能的未来",
    "深度学习是一种",
    "计算机科学",
    "Python 是",
]


def generate_samples(model, tokenizer, prompts=None, max_new_tokens=80, device="cuda"):
    """让模型生成文本，看它是否"学会说话"。"""
    import torch
    if prompts is None:
        prompts = PROMPTS
    model.eval()
    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"prompt": prompt, "generation": gen})
    return results


# ============================================================
# 评测报告
# ============================================================
def print_report(title, metrics, samples=None):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Loss:           {metrics['loss']:.4f}")
    print(f"  Perplexity:     {metrics['perplexity']:.2f}")
    if samples:
        print(f"\n  --- 生成样例 ---")
        for s in samples:
            print(f"\n  [prompt] {s['prompt']}")
            print(f"  [gen]    {s['generation'][:120]}")


# ============================================================
# 主入口
# ============================================================
def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="训后模型路径")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data-file", default=str(ROOT / "data" / "processed" / "token_blocks.npy"))
    ap.add_argument("--baseline", action="store_true", help="同时评测随机初始 baseline 做对比")
    ap.add_argument("--num-eval-blocks", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] 设备: {device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载评测数据
    blocks = np.load(args.data_file)
    eval_blocks = blocks[:args.num_eval_blocks]
    print(f"[info] 评测数据: {eval_blocks.shape}")

    results = {}

    # ---- Baseline（随机初始）----
    if args.baseline:
        print("\n[baseline] 构造随机初始化模型...")
        baseline_model = build_model(cfg["model"]["arch"]).to(device)
        params = count_parameters(baseline_model)
        print(f"    参数量: {params['total_M']:.1f}M（随机权重）")
        bm = compute_perplexity(baseline_model, eval_blocks, tokenizer, device=device)
        bs = generate_samples(baseline_model, tokenizer, device=device)
        print_report("Baseline（随机初始化）", bm, bs)
        results["baseline"] = bm

    # ---- 训后模型 ----
    if args.model:
        print(f"\n[model] 加载训后模型: {args.model}")
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                  device_map="auto", trust_remote_code=True)
        m = compute_perplexity(model, eval_blocks, tokenizer, device=device)
        s = generate_samples(model, tokenizer, device=device)
        print_report(f"训后模型: {args.model}", m, s)
        results["after_pretrain"] = m

        # 对比
        if args.baseline:
            print(f"\n{'='*60}")
            print("  预训练前 vs 后 对比")
            print(f"{'='*60}")
            print(f"  {'指标':<12} {'随机初始':>14} {'预训练后':>14} {'变化':>10}")
            for k in ["loss", "perplexity"]:
                b = results["baseline"][k]
                a = results["after_pretrain"][k]
                print(f"  {k:<12} {b:>14.2f} {a:>14.2f} {(a-b):>+10.2f}")
    else:
        # 没指定 model，只跑 baseline
        if not args.baseline:
            print("[fatal] 请指定 --model 或 --baseline")
            sys.exit(1)

    # 保存
    import json
    out_path = ROOT / "results" / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] 结果保存到 {out_path}")


if __name__ == "__main__":
    main()
