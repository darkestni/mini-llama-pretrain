#!/usr/bin/env bash
# ============================================================
# run_all.sh — 集群一键预训练流程
# 数据准备 → 预训练 → 评测
# ============================================================
set -e
cd "$(dirname "$0")/.."
echo "项目根目录: $(pwd)"

echo ""
echo "============================================================"
echo "  [0/4] 环境检查"
echo "============================================================"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

# ---- 1. 依赖 ----
if [ "$1" != "--no-install" ]; then
    echo ""
    echo "  [1/4] 安装依赖"
    pip install -r requirements.txt
fi

# ---- 2. 数据准备 ----
echo ""
echo "============================================================"
echo "  [2/4] 数据准备（Wikipedia）"
echo "============================================================"
# 联网则下载 wikipedia，否则用 sample
if python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)" 2>/dev/null; then
    echo "网络可用，下载 Wikipedia 中文子集..."
    python data/prepare_data.py --source wikipedia --language zh --num-samples 50000
else
    echo "[warn] 无法联网，降级为内置小样本（仅走通流程）"
    python data/prepare_data.py --source sample --num-samples 1000
fi

# ---- 3. 预训练 ----
echo ""
echo "============================================================"
echo "  [3/4] 预训练（从随机权重）"
echo "============================================================"
python train/pretrain.py --config configs/default.yaml

# ---- 4. 评测 ----
echo ""
echo "============================================================"
echo "  [4/4] 评测（困惑度 + 生成对比）"
echo "============================================================"
AFTER=$(python -c "import yaml; print(yaml.safe_load(open('configs/default.yaml'))['train']['output_dir'])")
python eval/evaluate.py --model "$AFTER" --baseline --data-file data/processed/token_blocks.npy

echo ""
echo "============================================================"
echo "  ✅ 全流程完成"
echo "============================================================"
echo "预训练产出: $AFTER"
echo "评测结果: results/eval_metrics.json"
echo "训练曲线: results/training_loss.png"
echo ""
echo "下一步：把训练 loss 曲线 + 生成对比填入 results/experiment_record.md"
