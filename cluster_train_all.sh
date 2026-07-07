#!/usr/bin/env bash
# ============================================================
# cluster_train_all.sh
# 在集群上一键跑三个训练项目（后台串行，断开 SSH 不停）
#
# 用法（登录集群后）：
#   cd ~
#   curl -o train_all.sh https://raw.githubusercontent.com/darkestni/mini-llama-pretrain/main/cluster_train_all.sh
#   bash train_all.sh 2>&1 | tee train_all.log
#
# 或者后台跑（推荐，断开 SSH 也不停）：
#   nohup bash train_all.sh > train_all.log 2>&1 &
#   tail -f train_all.log    # 看进度
# ============================================================
set -e

# ---- 配置 ----
WORKDIR="${HOME}/llm-train-projects"
LOGDIR="${WORKDIR}/logs"
mkdir -p "$WORKDIR" "$LOGDIR"

echo "============================================================"
echo "  集群训练一键启动 · $(date)"
echo "  工作目录: $WORKDIR"
echo "============================================================"
echo ""

# ---- 0. 检查 GPU ----
echo "[0/5] 检查环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "  GPU 数量: $GPU_COUNT"
else
    echo "  [警告] 未检测到 nvidia-smi，当前节点可能无 GPU"
    echo "  如果你需要申请计算节点，请先联系课题组或用调度系统"
    read -p "  确认继续？(y/N) " yn
    [ "$yn" != "y" ] && exit 1
fi

# ---- 1. Clone 三个项目 ----
echo ""
echo "[1/5] Clone 项目..."
cd "$WORKDIR"

clone_or_pull() {
    local url=$1
    local dir=$2
    if [ -d "$dir" ]; then
        echo "  $dir 已存在，git pull..."
        cd "$dir" && git pull && cd ..
    else
        git clone "$url" "$dir"
    fi
}

clone_or_pull "https://github.com/darkestni/mini-llama-pretrain.git" "mini-llama-pretrain"
clone_or_pull "https://github.com/darkestni/nlp.git" "sarcasm-r1-lite"
clone_or_pull "https://github.com/darkestni/code-agent-r1.git" "code-agent-r1"

# ---- 2. 创建 conda 环境 ----
echo ""
echo "[2/5] 配置 conda 环境..."
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "llm-train"; then
        echo "  环境 llm-train 已存在"
    else
        echo "  创建环境 llm-train..."
        conda create -n llm-train python=3.10 -y
    fi
    conda activate llm-train
else
    echo "  [警告] 未找到 conda，使用系统 python"
fi

# ---- 3. 装依赖（三个项目共用一个环境）----
echo ""
echo "[3/5] 安装依赖..."
pip install --upgrade pip
pip install torch transformers accelerate trl datasets tokenizers
pip install pyyaml tqdm numpy scikit-learn matplotlib pandas

echo "  验证 torch + GPU:"
python -c "import torch; print(f'  torch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

# ---- 4. 定义训练任务 ----
echo ""
echo "[4/5] 启动训练（串行：一个跑完跑下一个）..."

# 任务 1：预训练（最重要，先跑）
echo ""
echo "============================================================"
echo "  [训练 1/3] MiniLlama 预训练"
echo "============================================================"
cd "$WORKDIR/mini-llama-pretrain"
# 先准备数据（联网下 wikipedia，失败则用 sample）
python data/prepare_data.py --source wikipedia --language zh --num-samples 50000 2>&1 | tee "$LOGDIR/pretrain_data.log" || \
    python data/prepare_data.py --source sample --num-samples 5000
# 训练（先 10 步验证，再正式）
python train/pretrain.py --config configs/default.yaml --max-steps 10 2>&1 | tee "$LOGDIR/pretrain_smoke.log"
echo "  [smoke test 通过，开始正式训练]"
python train/pretrain.py --config configs/default.yaml 2>&1 | tee "$LOGDIR/pretrain_full.log"
# 评测
python eval/evaluate.py --model outputs/pretrain_run --baseline 2>&1 | tee "$LOGDIR/pretrain_eval.log"
echo "  ✅ 预训练完成"

# 任务 2：Sarcasm GRPO
echo ""
echo "============================================================"
echo "  [训练 2/3] Sarcasm-R1-Lite GRPO"
echo "============================================================"
cd "$WORKDIR/sarcasm-r1-lite"
bash scripts/run_all.sh 2>&1 | tee "$LOGDIR/sarcasm_full.log"
echo "  ✅ Sarcasm 训练完成"

# 任务 3：Code Agent GRPO
echo ""
echo "============================================================"
echo "  [训练 3/3] CodeAgent-R1 GRPO"
echo "============================================================"
cd "$WORKDIR/code-agent-r1"
bash scripts/run_all.sh 2>&1 | tee "$LOGDIR/codeagent_full.log"
echo "  ✅ Code Agent 训练完成"

# ---- 5. 汇总 ----
echo ""
echo "============================================================"
echo "  ✅ 全部训练完成 · $(date)"
echo "============================================================"
echo ""
echo "结果文件位置："
echo "  预训练:  $WORKDIR/mini-llama-pretrain/results/"
echo "           - training_loss.png (loss 曲线)"
echo "           - eval_metrics.json (困惑度)"
echo "  Sarcasm: $WORKDIR/sarcasm-r1-lite/results/"
echo "           - baseline_metrics.json + after_metrics.json"
echo "  Code Agent: $WORKDIR/code-agent-r1/results/"
echo ""
echo "把这些结果 scp 回本地，填进简历的 __ 占位符。"
echo ""
echo "下载命令（在本地 Mac 跑）："
echo "  scp -P 10022 -r cse12312606@172.18.34.26:$WORKDIR/mini-llama-pretrain/results/ ~/Desktop/预训练结果/"
