#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=130G
#SBATCH --time=00:10:00
#SBATCH --job-name=seekui_3gpu_stage1
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/3gpu_stage1_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/3gpu_stage1_%j.err

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
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# 4. 进入工作目录
cd /uufs/chpc.utah.edu/common/home/u1442451/SeekUI

# 5. 打印调度到的 GPU 硬件信息
echo "=== Scheduled GPUs ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 6. 运行3GPU训练
bash stage1/qwen-vl-finetune/run_eye_tracking_3gpu.sh