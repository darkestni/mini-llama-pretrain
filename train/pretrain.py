"""
pretrain.py
===========
MiniLlama 预训练主脚本：从随机权重开始，做因果语言建模（CLM）训练。

⚠️ 需要 GPU + torch + transformers。
   Mac 上只做前向验证，不实际训练。
   集群上：python train/pretrain.py --config configs/default.yaml

核心流程：
    1. 构造随机初始化的 MiniLlama（111M）
    2. 加载 tokenize 好的 wikipedia token 块
    3. HF Trainer 做因果语言建模训练
    4. 保存 checkpoint + 训练 loss 曲线
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "data"))

import yaml
import numpy as np

from config import build_model, count_parameters  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    import torch  # 延迟 import，让 Mac 上没 torch 也能 --help
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data-file", default=str(ROOT / "data" / "processed" / "token_blocks.npy"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 config 的 max_steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    seed = cfg["run"]["seed"]

    print("=" * 60)
    print("MiniLlama 预训练")
    print("=" * 60)

    # ---- 1. 构造随机初始化模型 ----
    print("\n[1/4] 构造随机初始化的 MiniLlama...")
    model = build_model(model_cfg["arch"])
    params = count_parameters(model)
    print(f"    参数量: {params['total_M']:.1f}M")
    print(f"    架构: {model.config.num_hidden_layers}L × {model.config.hidden_size}D")
    print(f"    初始化: 随机权重（从零预训练，非微调）")

    if args.smoke_test:
        print("\n[smoke-test] 环境验证通过，跳过训练。")
        # 前向测试
        dummy = torch.randint(0, model_cfg["arch"]["vocab_size"], (2, 64))
        with torch.no_grad():
            out = model(dummy, labels=dummy)
        print(f"    随机初始 loss: {out.loss.item():.2f}（应 ~10-12）")
        return

    # ---- 2. 加载数据 ----
    print("\n[2/4] 加载预训练数据...")
    from datasets import Dataset
    blocks = np.load(args.data_file)
    print(f"    数据: {blocks.shape}（{blocks.shape[0]} 个 {blocks.shape[1]}-token 块）")
    # 高效构建：直接用 numpy 数组，不转 list（tolist 会爆内存）
    attention = np.ones_like(blocks, dtype=np.int64)
    dataset = Dataset.from_dict({
        "input_ids": blocks,
        "labels": blocks,
        "attention_mask": attention,
    })
    dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    print(f"    样本数: {len(dataset)}")

    # ---- 3. 配置训练 ----
    print("\n[3/4] 配置 Trainer...")
    from transformers import TrainingArguments, Trainer, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = args.output_dir or train_cfg["output_dir"]
    max_steps = args.max_steps or train_cfg["max_steps"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        bf16=train_cfg["bf16"] and torch.cuda.is_available(),
        fp16=train_cfg["fp16"] and torch.cuda.is_available() and not train_cfg["bf16"],
        seed=seed,
        report_to="none",
        dataloader_num_workers=4,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ---- 4. 训练 ----
    print(f"\n[4/4] 开始预训练（{max_steps} 步）...")
    print(f"    等效 batch: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"    学习率: {train_cfg['learning_rate']}")
    print(f"    输出: {output_dir}")

    train_result = trainer.train()

    # 保存
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存训练日志
    log_history = trainer.state.log_history
    import json
    with open(ROOT / "results" / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\n[done] 预训练完成，保存到 {output_dir}")
    print(f"    最终 loss: {train_result.training_loss:.4f}")
    print(f"\n下一步：python eval/evaluate.py --model {output_dir}")

    # 绘制 loss 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [x["step"] for x in log_history if "loss" in x]
        losses = [x["loss"] for x in log_history if "loss" in x]
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("MiniLlama Pretraining Loss")
        plt.savefig(ROOT / "results" / "training_loss.png", dpi=120, bbox_inches="tight")
        print(f"    Loss 曲线: results/training_loss.png")
    except Exception as e:
        print(f"    [warn] loss 曲线绘制失败: {e}")


if __name__ == "__main__":
    main()
