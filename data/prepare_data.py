"""
prepare_data.py
===============
预训练数据准备：从 Wikipedia 下载文本 → tokenize → 打包成固定长度块。

预训练数据的关键特征：
    - 无标注（纯文本，不是指令对）
    - 大规模
    - 打包成等长 token 块（不像 SFT 有变长序列）

三条来源（自动降级）：
    1. HuggingFace wikipedia 数据集（集群联网用）
    2. 本地 raw 文本（若 data/raw/ 有）
    3. 内置小样本（离线走通用）

用法：
    python data/prepare_data.py --source wikipedia --language zh --num-samples 50000
    python data/prepare_data.py --source sample                # 离线走通
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# 路径 3：内置小样本（离线可跑）
# ============================================================
SAMPLE_TEXTS = [
    "机器学习是人工智能的一个分支，它研究如何让计算机系统从数据中学习。",
    "深度学习使用多层神经网络来模拟人脑的学习过程，在图像识别和自然语言处理等领域取得了显著进展。",
    "transformer 架构由注意力机制组成，它摒弃了循环神经网络的顺序计算，实现了高度并行化。",
    "预训练语言模型首先在大规模无标注文本上进行语言建模，然后通过微调适应下游任务。",
    "Python 是一种广泛使用的高级编程语言，以代码可读性和简洁的语法著称。",
    "南方科技大学是一所位于深圳的研究型大学，致力于培养具有创新精神的科技人才。",
    "强化学习通过与环境交互获取奖励信号，学习最优的决策策略。",
    "数据结构是计算机存储和组织数据的方式，常见的有数组、链表、树和图等。",
    "操作系统管理计算机硬件资源，为应用程序提供运行环境和服务。",
    "数据库系统用于存储、管理和查询大量结构化数据，支持事务和并发控制。",
    "计算机视觉研究如何让机器理解和处理图像信息，包括识别、检测和分割等任务。",
    "自然语言处理致力于让计算机理解、生成和处理人类语言。",
    "云计算通过互联网提供计算资源和服务，包括基础设施、平台和软件三种模式。",
    "区块链是一种分布式账本技术，通过密码学保证数据的不可篡改性。",
    "量子计算利用量子力学的叠加和纠缠原理，在某些问题上具有指数级加速能力。",
]


def load_sample_data(num_samples: int = 1000):
    """内置小样本：复制扩充。"""
    out = []
    while len(out) < num_samples:
        out.extend(SAMPLE_TEXTS)
    return out[:num_samples]


# ============================================================
# 路径 1：HuggingFace wikipedia
# ============================================================
def load_wikipedia(language: str, num_samples, cache_dir: str):
    """从 HuggingFace 下载 wikipedia 数据集。"""
    try:
        from datasets import load_dataset
        # wikipedia 现在叫 wikimedia/wikipedia
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{language}",
                          split="train", cache_dir=cache_dir)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        # wikipedia 的 text 字段
        text_field = "text"
        return [ex[text_field] for ex in ds if ex.get(text_field)]
    except Exception as e:
        print(f"[warn] wikipedia 加载失败: {e}")
        print("[hint] 可能是版本问题，尝试旧版 wikipedia loader...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikipedia", f"20220301.{language}",
                              split="train", cache_dir=cache_dir, trust_remote_code=True)
            if num_samples:
                ds = ds.select(range(min(num_samples, len(ds))))
            return [ex["text"] for ex in ds if ex.get("text")]
        except Exception as e2:
            print(f"[warn] 旧版也失败: {e2}")
            return None


def load_local_raw():
    """从 data/raw/ 读本地文本文件。"""
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        return None
    texts = []
    for f in raw_dir.glob("*.txt"):
        texts.append(f.read_text(encoding="utf-8"))
    for f in raw_dir.glob("*.jsonl"):
        import json
        with open(f) as fp:
            for line in fp:
                obj = json.loads(line)
                if "text" in obj:
                    texts.append(obj["text"])
    return texts if texts else None


# ============================================================
# Tokenize + 打包（核心）
# ============================================================
def tokenize_and_pack(texts, tokenizer, max_length: int = 512):
    """
    将文本列表 tokenize 后，拼接并切分成固定长度的 token 块。

    这是预训练数据准备的经典做法：
        1. 把所有文本 tokenize 后拼接成一长串
        2. 按 max_length 切分成等长块
        （不像 SFT 一条一条 padding）

    Returns:
        input_ids: np.ndarray, shape [N, max_length]
    """
    print(f"[tokenize] 处理 {len(texts)} 段文本...")

    # 方式一：用 tokenizer 的 batch 编码（快）
    all_ids = []
    from tqdm import tqdm
    batch_size = 1000
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, add_special_tokens=False, truncation=False)
        for ids in enc["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)  # 文档之间用 eos 分隔

    print(f"[tokenize] 总 token 数: {len(all_ids):,}")

    # 切分成固定长度块
    total_tokens = len(all_ids)
    num_blocks = total_tokens // max_length
    # 截断到整数倍
    all_ids = all_ids[:num_blocks * max_length]
    blocks = np.array(all_ids, dtype=np.int32).reshape(-1, max_length)

    print(f"[pack] 切分成 {num_blocks} 个 {max_length} 长度的块")
    return blocks


def save_blocks(blocks, output_dir: Path):
    """保存打包好的 token 块。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    # 存成 .npy（高效）
    out_path = output_dir / "token_blocks.npy"
    np.save(out_path, blocks)
    print(f"[save] {blocks.shape} → {out_path}")

    # 也存一个样本预览（前 5 块 decode）
    from transformers import AutoTokenizer
    return out_path


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample",
                    choices=["sample", "wikipedia", "local"])
    ap.add_argument("--language", default="zh")
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()

    print("=" * 60)
    print(f"MiniLlama 数据准备 · source={args.source}")
    print("=" * 60)

    # 1. 加载文本
    print(f"\n[1/3] 加载文本数据...")
    if args.source == "sample":
        texts = load_sample_data(args.num_samples)
        print(f"    内置小样本: {len(texts)} 段（离线走通用，不可真训）")
    elif args.source == "wikipedia":
        texts = load_wikipedia(args.language, args.num_samples, str(CACHE_DIR))
        if texts is None:
            print("[fallback] wikipedia 加载失败，降级为内置小样本")
            texts = load_sample_data(args.num_samples)
    elif args.source == "local":
        texts = load_local_raw()
        if texts is None:
            print("[fatal] data/raw/ 下无可用文件")
            sys.exit(1)

    # 2. 加载 tokenizer
    print(f"\n[2/3] 加载 tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenize + 打包
    print(f"\n[3/3] Tokenize 并打包成 {args.max_length} 长度的块...")
    blocks = tokenize_and_pack(texts, tokenizer, args.max_length)

    # 保存
    save_blocks(blocks, PROCESSED_DIR)

    # 预览
    print(f"\n[预览] 第 1 块的 decode（前 100 字）:")
    print(tokenizer.decode(blocks[0], skip_special_tokens=True)[:100])

    print(f"\n✅ 数据准备完成，可送入预训练")


if __name__ == "__main__":
    main()
