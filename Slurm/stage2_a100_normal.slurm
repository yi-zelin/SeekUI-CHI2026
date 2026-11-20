#!/usr/bin/env bash
#SBATCH --job-name=seekui-stage2-a100-normal
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --qos=soc-gpushort-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/stage2_a100_normal_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/stage2_a100_normal_%j.err

set -euo pipefail

REPO_ROOT=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI
STAGE2_ROOT="${REPO_ROOT}/stage2/src/seekui"
SCRATCH_ROOT=/scratch/general/vast/u1442451
export MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/SeekUI/data/SeekUI_sft}"
export DATA_PATH="${DATA_PATH:-${SCRATCH_ROOT}/SeekUI/data/vsgui_train_seekui_qwen2_5_explanation}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/SeekUI/runs/stage2_a100_normal}"
OUTPUT_DIR="${RUN_ROOT}/job_${SLURM_JOB_ID}"
export OUTPUT_DIR

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_ROOT}/huggingface" "${SCRATCH_ROOT}/torch_cache" "${SCRATCH_ROOT}/triton_cache"

module load cuda/12.4.0
eval "$(/uufs/chpc.utah.edu/common/home/u1442451/miniconda3/bin/conda shell.bash hook)"
conda activate seekui

export PYTHONUNBUFFERED=1
export PYTHONPATH="${STAGE2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/torch_cache"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/triton_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME='^docker0,lo'
export DEBUG_MODE=true
export LOG_PATH="${OUTPUT_DIR}/reward_debug.log"

cd "${STAGE2_ROOT}"

echo "=== Job and GPU ==="
echo "job_id=${SLURM_JOB_ID} node=$(hostname) start=$(date --iso-8601=seconds)"
echo "model_path=${MODEL_PATH}"
echo "data_path=${DATA_PATH}"
echo "output_dir=${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Scratch is periodically cleaned. Restore incomplete model/data snapshots before
# starting the 20-minute training test, then fail early if they remain unusable.
python -u - <<'PY'
import json
import os
from pathlib import Path

from datasets import DatasetDict
from huggingface_hub import snapshot_download

model_path = Path(os.environ["MODEL_PATH"])
data_path = Path(os.environ["DATA_PATH"])


def model_complete(path):
    required = ["config.json", "preprocessor_config.json", "tokenizer_config.json"]
    if not all((path / name).is_file() for name in required):
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        shards = set(index.get("weight_map", {}).values())
        return bool(shards) and all(
            (path / shard).is_file() and (path / shard).stat().st_size > 0
            for shard in shards
        )
    return any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").is_file()


if not model_complete(model_path):
    print(f"Model missing or incomplete; restoring sushizixin1/SeekUI_sft into {model_path}", flush=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="sushizixin1/SeekUI_sft",
        repo_type="model",
        local_dir=str(model_path),
    )
if not model_complete(model_path):
    raise RuntimeError(f"Model is still incomplete after download: {model_path}")
print("Model snapshot validation passed", flush=True)


def dataset_complete(path):
    if not (path / "dataset_dict.json").is_file():
        return False
    dataset = DatasetDict.load_from_disk(str(path))
    if "train" not in dataset or len(dataset["train"]) == 0:
        return False
    image = dataset["train"][0]["image"]
    return getattr(image, "width", 0) > 0 and getattr(image, "height", 0) > 0


try:
    data_ready = dataset_complete(data_path)
except Exception as error:
    print(f"Dataset validation failed before download: {error}", flush=True)
    data_ready = False
if not data_ready:
    print(
        "Dataset missing or unreadable; restoring "
        f"sushizixin1/vsgui_train_seekui_qwen2_5_explanation into {data_path}",
        flush=True,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="sushizixin1/vsgui_train_seekui_qwen2_5_explanation",
        repo_type="dataset",
        local_dir=str(data_path),
    )
if not dataset_complete(data_path):
    raise RuntimeError(f"Dataset is still unreadable after download: {data_path}")
print("Dataset and embedded-image validation passed", flush=True)
PY

# Run the unabridged 30-epoch training loop until Slurm reaches its 20-minute
# limit. No step cap or checkpoint saving is used. GPU samples provide data for
# estimating steady-state throughput and memory requirements.
GPU_MONITOR_LOG="${OUTPUT_DIR}/gpu_usage.csv"
echo "timestamp,name,utilization_gpu_percent,memory_used_mib,memory_total_mib" > "${GPU_MONITOR_LOG}"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits >> "${GPU_MONITOR_LOG}"
        sleep 10
    done
) &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

srun python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    src/open_r1/grpo.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATA_PATH}" \
    --deepspeed ./local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 30 \
    --run_name "seekui-stage2-a100-normal-${SLURM_JOB_ID}" \
    --save_strategy no \
    --num_generations 4 \
    --max_completion_length 512 \
    --multi_reference_scanpath true \
    --counterfactual_scanpath true \
    --counterfactual_reward_weight 0.1 \
    --counterfactual_margin 0.0 \
    --counterfactual_human_gate true

echo "=== Training finished before the time limit: $(date --iso-8601=seconds) ==="
