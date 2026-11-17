#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/general/vast/u1442451/huggingface
export TORCH_HOME=/scratch/general/vast/u1442451/torch_cache

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR="/scratch/general/vast/u1442451/huggingface/hub"
DATASETS="vsgui_text%100"
OUTPUT_DIR="/scratch/general/vast/u1442451/SeekUI_sft"

# Check if the dataset image folder is empty (scratch dirs may be purged); if so, re-extract it
DATA_DIR="/scratch/general/vast/u1442451/SeekUI/data/vsgui10k-images"
DATA_ZIP="/scratch/general/vast/u1442451/SeekUI/data/vsgui10k-images-20260908T180049Z-1-001.zip"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR")" ]; then
    echo "Data folder $DATA_DIR is empty or missing. Removing it and extracting $DATA_ZIP ..."
    rm -rf "$DATA_DIR"
    unzip -o -q "$DATA_ZIP" -d "$(dirname "$DATA_DIR")" || { echo "ERROR: failed to extract $DATA_ZIP"; exit 1; }
    echo "Extraction complete."
fi

# Change to script directory to ensure correct relative paths
cd "$SCRIPT_DIR"

# Multi-GPU training with 2 GPUs
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=4622 \
qwenvl/train/train_qwen.py \
--model_name_or_path $MODEL_PATH \
--tune_mm_llm True \
--tune_mm_vision False \
--tune_mm_mlp True \
--dataset_use $DATASETS \
--output_dir $OUTPUT_DIR \
--cache_dir $CACHE_DIR \
--bf16 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-7 \
--mm_projector_lr 1e-5 \
--vision_tower_lr 1e-6 \
--optim adamw_torch \
--model_max_length 4096 \
--data_flatten True \
--data_packing True \
--max_pixels 451584 \
--min_pixels 12544 \
--base_interval 2 \
--video_max_frames 8 \
--video_min_frames 4 \
--video_max_frame_pixels 1304576 \
--video_min_frame_pixels 200704 \
--num_train_epochs 200 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--weight_decay 0.01 \
--logging_steps 10 \
--save_steps 2000 \
--save_total_limit 3 \
--dataloader_num_workers 4 \
--deepspeed ./scripts/zero3.json