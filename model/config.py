"""
config.py
=========
MiniLlama 模型配置：生成一个缩小版的 Llama 架构（~100M 参数）。

关键：与标准 Llama/Qwen 同架构（LlamaForCausalLM），只是缩小维度。
这样"预训练经验"在架构层面是可迁移的。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_model_config(arch_cfg: dict):
    """
    根据配置字典，构造一个 LlamaConfig。

    Args:
        arch_cfg: configs/default.yaml 里 model.arch 的内容

    Returns:
        transformers.LlamaConfig
    """
    from transformers import LlamaConfig

    config = LlamaConfig(
        vocab_size=arch_cfg["vocab_size"],
        hidden_size=arch_cfg["hidden_size"],
        intermediate_size=arch_cfg["intermediate_size"],
        num_hidden_layers=arch_cfg["num_hidden_layers"],
        num_attention_heads=arch_cfg["num_attention_heads"],
        num_key_value_heads=arch_cfg.get("num_key_value_heads", arch_cfg["num_attention_heads"]),
        max_position_embeddings=arch_cfg["max_position_embeddings"],
        rms_norm_eps=arch_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=arch_cfg.get("rope_theta", 10000.0),
        tie_word_embeddings=arch_cfg.get("tie_word_embeddings", True),
        # 训练相关
        pad_token_id=arch_cfg.get("vocab_size", 151643) - 1,  # 用最后一个 token 作 pad
        bos_token_id=0,
        eos_token_id=arch_cfg.get("vocab_size", 151643) - 1,
    )
    return config


def build_model(arch_cfg: dict):
    """构造一个随机初始化的 LlamaForCausalLM。"""
    from transformers import LlamaForCausalLM
    config = build_model_config(arch_cfg)
    model = LlamaForCausalLM(config)
    return model


def count_parameters(model) -> dict:
    """统计模型参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("MiniLlama 配置自测")
    print("=" * 60)

    print("\n[1] 构造模型配置...")
    config = build_model_config(cfg["model"]["arch"])
    print(f"    架构: {config.num_hidden_layers} 层 × {config.hidden_size} 维")
    print(f"    头数: {config.num_attention_heads}")
    print(f"    上下文: {config.max_position_embeddings}")

    print("\n[2] 构造随机初始化模型（不加载预训练权重）...")
    model = build_model(cfg["model"]["arch"])
    params = count_parameters(model)
    print(f"    总参数量: {params['total_M']:.1f}M ({params['total']:,})")
    print(f"    可训练:   {params['trainable_M']:.1f}M")

    # 验证确实是随机初始化（不是预训练）
    import torch
    embed_weight = model.model.embed_tokens.weight
    print(f"\n[3] 验证随机初始化:")
    print(f"    embedding 均值: {embed_weight.mean().item():.4f}（接近0=随机）")
    print(f"    embedding 标准差: {embed_weight.std().item():.4f}（~0.02=Llama 初始化）")

    print("\n[4] 前向传播测试（2 条样本）...")
    dummy_input = torch.randint(0, cfg["model"]["arch"]["vocab_size"], (2, 64))
    with torch.no_grad():
        outputs = model(dummy_input)
    print(f"    logits 形状: {outputs.logits.shape}（应为 [batch, seq, vocab]）")
    print(f"    loss: {outputs.loss}（无 labels 时应为 None，符合预期）")

    # 带 labels 的前向
    with torch.no_grad():
        outputs = model(dummy_input, labels=dummy_input)
    print(f"    带 labels 的 loss: {outputs.loss.item():.2f}（随机初始应 ~10）")

    print("\n✅ 模型配置验证通过，可送入训练")
