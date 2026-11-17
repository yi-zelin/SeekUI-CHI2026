#!/usr/bin/env bash
#SBATCH --job-name=h0083-tc-same-llm2
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/test_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/test_%j.err



# 1. 确保日志与缓存路径存在
mkdir -p /uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs
mkdir -p /scratch/general/vast/u1442451/triton_cache
mkdir -p /scratch/general/vast/u1442451/huggingface

# 2. 加载模块与 Conda 环境
module load cuda/11.8
eval "$(/uufs/chpc.utah.edu/common/home/u1442451/miniconda3/bin/conda shell.bash hook)"
conda activate seekui

# 3. 环境变量（避免 NFS Triton 卡顿与 WandB 阻断）
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/general/vast/u1442451/huggingface
export TORCH_HOME=/scratch/general/vast/u1442451/torch_cache
export TRITON_CACHE_DIR=/scratch/general/vast/u1442451/triton_cache
export WANDB_MODE=offline

# 4. 进入工作目录
cd /uufs/chpc.utah.edu/common/home/u1442451/SeekUI

# 5. 打印调度到的 GPU 硬件信息
echo "=== Scheduled GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 6. 运行训练测试
bash stage1/qwen-vl-finetune/run_eye_tracking_2gpu.sh
