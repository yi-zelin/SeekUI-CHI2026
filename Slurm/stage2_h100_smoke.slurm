#!/usr/bin/env bash
#SBATCH --job-name=seekui-stage2-h100-smoke
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --qos=soc-gpushort-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/stage2_h100_smoke_%j.out
#SBATCH --error=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI/logs/stage2_h100_smoke_%j.err

set -euo pipefail

REPO_ROOT=/uufs/chpc.utah.edu/common/home/u1442451/SeekUI
STAGE2_ROOT="${REPO_ROOT}/stage2/src/seekui"
SCRATCH_ROOT=/scratch/general/vast/u1442451
export MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/SeekUI/data/SeekUI_sft}"
export DATA_PATH="${DATA_PATH:-${SCRATCH_ROOT}/SeekUI/data/vsgui_train_seekui_qwen2_5_explanation}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/SeekUI/runs/stage2_h100_smoke}"
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

# One real GRPO optimizer step: two sampled completions, reward calculation,
# full-parameter backward propagation, checkpoint-1, and a final model save.
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
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --max_steps 1 \
    --num_train_epochs 1 \
    --run_name "seekui-stage2-h100-smoke-${SLURM_JOB_ID}" \
    --save_strategy steps \
    --save_steps 1 \
    --save_only_model true \
    --save_total_limit 1 \
    --num_generations 2 \
    --max_completion_length 512 \
    --multi_reference_scanpath true \
    --counterfactual_scanpath true \
    --counterfactual_reward_weight 0.1 \
    --counterfactual_margin 0.0 \
    --counterfactual_human_gate true

python -u - <<'PY'
import json
import math
import os
import re
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
checkpoint_dir = output_dir / "checkpoint-1"


def validate_model_directory(path):
    if not (path / "config.json").is_file():
        raise RuntimeError(f"Missing config.json in {path}")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            shards = set(json.load(handle)["weight_map"].values())
        missing = [shard for shard in shards if not (path / shard).is_file()]
        if missing:
            raise RuntimeError(f"Missing saved model shards in {path}: {missing}")
        total_bytes = sum((path / shard).stat().st_size for shard in shards)
    elif (path / "model.safetensors").is_file():
        total_bytes = (path / "model.safetensors").stat().st_size
    else:
        raise RuntimeError(f"No safetensors model found in {path}")
    if total_bytes < 1_000_000_000:
        raise RuntimeError(f"Saved model is unexpectedly small ({total_bytes} bytes): {path}")
    print(f"Saved model validation passed: {path} ({total_bytes / 1e9:.2f} GB)")


validate_model_directory(checkpoint_dir)
validate_model_directory(output_dir)
state_path = checkpoint_dir / "trainer_state.json"
with state_path.open(encoding="utf-8") as handle:
    state = json.load(handle)
if state.get("global_step") != 1:
    raise RuntimeError(f"Expected global_step=1, found {state.get('global_step')}")
if not state.get("log_history"):
    raise RuntimeError("trainer_state.json contains no training metrics")
metrics = next(
    (entry for entry in reversed(state["log_history"]) if entry.get("step") == 1),
    None,
)
if metrics is None:
    raise RuntimeError("No step-1 training metrics found")
for name in ["loss", "grad_norm", "reward"]:
    if name not in metrics or not math.isfinite(metrics[name]):
        raise RuntimeError(f"Missing or non-finite training metric: {name}={metrics.get(name)}")
if metrics["grad_norm"] <= 0:
    raise RuntimeError(f"Smoke test produced no non-zero gradient: {metrics}")

reward_log_path = Path(os.environ["LOG_PATH"])
reward_log = reward_log_path.read_text(encoding="utf-8")
if not re.search(r"student_answer_bbox: \[\[", reward_log):
    raise RuntimeError("No generated completion contained a parseable scanpath")
reliability_values = re.findall(r"counterfactual_human_reliability: ([^\n]+)", reward_log)
if not reliability_values or all(value == "None" for value in reliability_values):
    raise RuntimeError("The counterfactual human-separability gate was not evaluated")

print(f"Training state validation passed: global_step={state['global_step']}")
print(f"Step-1 metrics: {metrics}")
print(f"Human-separability values: {reliability_values}")
print(f"Reward debug log: {reward_log_path}")
print("STAGE2_H100_SMOKE_TEST_PASSED")
PY

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "=== Smoke test completed: $(date --iso-8601=seconds) ==="
