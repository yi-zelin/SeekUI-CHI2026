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

# Change to script directory to ensure correct relative paths
cd "$SCRIPT_DIR"

# bs, lr /4, 开bf16, 其他参数保持不变
python -m torch.distributed.run --nproc_per_node=1 --master_port=4622 \
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
--deepspeed ./scripts/zero3.json \
