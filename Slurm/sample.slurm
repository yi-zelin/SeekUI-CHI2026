#!/usr/bin/env bash
#SBATCH --job-name=h0083-tc-same-llm2
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/general/vast/u6076267/seekui/logs/%x-%j.out
#SBATCH --error=/scratch/general/vast/u6076267/seekui/logs/%x-%j.err

set -euo pipefail

export SEEKUI_REPO="${SEEKUI_REPO:-$HOME/projects/SeekUI-CHI2026}"
export SEEKUI_WORK="${SEEKUI_WORK:-/scratch/general/vast/u6076267/seekui}"
export DATA_PATH="${DATA_PATH:-$SEEKUI_WORK/data}"
export MODEL_NAME="${MODEL_NAME:-SeekUI}"
export SPLIT_JSON="$SEEKUI_WORK/outputs/native_vsgui10k/processed_v2/splits/native_v2_unambiguous_text_color.json"
export ANSWER_MODE="scanpath_only"
export STATUS_FILTER="present"
export RUN_NAME="SeekUI_h0083_unambig_text_color_scanpath_present_seed123_llm2_lr3e7"
export DATASET_NAME="scanpath_metric_sft"
export SEED="123"
export TEST_FRAC="0.2"
export MAX_POINTS="24"
export TRAIN_SAMPLING="all"
export TUNE_MM_LLM="True"
export TUNE_MM_LLM_LAST_LAYERS="2"
export TUNE_MM_VISION="False"
export TUNE_MM_MLP="True"
export LEARNING_RATE="3e-7"
export MM_PROJECTOR_LR="5e-6"
export NUM_TRAIN_EPOCHS="2"
export TRAIN_BATCH_SIZE="1"
export GRAD_ACCUM="8"
export MASTER_PORT="4683"
export INFERENCE_PROMPT_MODE="scanpath_only"
export MAX_METRIC_FIXATIONS="24"
export SKIP_SS="1"

mkdir -p "$SEEKUI_WORK/logs"
cd "$SEEKUI_REPO"

bash scripts_utah/train_scanpath_metric_sft.slurm
