# MiniLlama-Pretrain

> 从随机权重预训练一个 Llama 架构的小型语言模型——真正的"从零预训练"实践。
> 111.3M 参数 / Llama 架构 / Wikipedia 数据 / 因果语言建模。

---

## 为什么做这个项目

大多数 LLM 学习者只会"微调"(拿现成模型做 SFT/RLHF),但**从未经历过真正的"预训练"**——从随机权重开始,让模型在海量无标注文本上学会"说话"。

这个项目补齐了这最后一块:从 `torch.nn.init` 的随机权重出发,经过数万步训练,看着模型的 loss 从 ~10(随机)降到 ~3(会说话),最终能生成连贯的中文/英文文本。

**这才是 JD 里"预训练"三个字真正的意思。**

---

## 方法

### 模型架构(Llama-style,缩小版)

| 维度 | 标准 Llama-7B | 本项目 MiniLlama |
|------|--------------|-----------------|
| 层数 | 32 | **6** |
| 隐藏维度 | 4096 | **576** |
| 注意力头数 | 32 | **9** |
| 参数量 | 7B | **111.3M** |
| 上下文长度 | 4096 | **512** |
| 架构 | LlamaForCausalLM | **LlamaForCausalLM**(同架构) |

同一个 Llama 架构,只是缩小到单卡能训。**架构相同 = 经验可迁移。**

### 数据(真实预训练数据)

- **来源**:HuggingFace `wikipedia/wikipedia`(中文 `zh` 或英文 `en` 子集)
- **特点**:**无标注**(这正是预训练数据的定义——不是 SFT 的指令对,不是 RLHF 的偏好对,就是纯文本)
- **处理**:原始文章 → 分词 → 打包成固定长度(512)的 token 块

### 训练目标

**因果语言建模(Causal Language Modeling, CLM)**:
预测序列中每个位置的下一个 token。这就是 GPT/Llama/Qwen 预训练用的同一个目标。

---

## 目录结构

```
mini-llama-pretrain/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml           ← 模型/数据/训练超参
├── model/
│   └── config.py              ← MiniLlama 配置生成(缩小版 Llama)
├── data/
│   ├── prepare_data.py        ← Wikipedia 下载 + tokenize + 打包
│   └── sample_data.py         ← 内置小样本(离线走通用)
├── train/
│   └── pretrain.py            ← 预训练主脚本(HF Trainer)
├── eval/
│   └── evaluate.py            ← 困惑度 + 文本生成评测
├── scripts/
│   ├── run_all.sh             ← 集群一键启动
│   └── run_local_small.sh     ← Mac 本地小规模走通
└── results/
    └── experiment_template.md
```

---

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

需要 Python 3.10+,GPU 训练需 CUDA + ≥10GB 显存(100M 模型)。

### 本地走通(Mac / CPU)

```bash
bash scripts/run_local_small.sh
```

验证:数据流程 + 模型配置 + 训练前向传播 + 评测 全链路通。**不真训**。

### 集群训练(GPU)

```bash
bash scripts/run_all.sh
```

依次:Wikipedia 下载 → tokenize → 预训练 → 困惑度评测 → 生成对比。

---

## 实验记录(已完成 · 2026-07 集群实跑)

| 指标 | 随机初始化 | 预训练后(留出集) | 训练终值 |
|------|----------|---------|---------|
| Loss | 12.04 | **3.44** | 2.93 |
| 困惑度 PPL | 168,966 | **31.34** | — |
| 生成质量 | 乱码 | **连贯中文**(语法正确、主题相关,事实性有限符合 111M 预期) | — |

配置:50,000 步 · batch 64 · lr 3e-4(cosine + warmup 5%)· bf16 · L40 单卡 10h32m · Wikipedia 186,989 个 512-token 块(9,570 万 token,约 17.1 epoch)。初始 loss ≈ ln(词表) ≈ 11.9,与实测 12.04 吻合。原始日志:`results/train_log.json`(1000 点)、`results/eval_metrics.json`;完整记录与 case study 见 `results/experiment_record.md`,曲线见 `results/figures/loss_curve.png`。

---

## 关于本项目

- 本项目是 **赵勋(@darkestni)的个人独立项目**。
- 全部代码(模型配置、数据流程、训练评测)由赵勋独立编写。
- 模型架构使用 HuggingFace `transformers` 的 `LlamaForCausalLM`。
- 若引用,请引用为:*赵勋, MiniLlama-Pretrain, GitHub, 2026.*

## License

MIT
