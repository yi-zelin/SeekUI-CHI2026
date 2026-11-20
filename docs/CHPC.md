# SeekUI 在 University of Utah CHPC 上的使用教程

本文介绍如何在 University of Utah 的 CHPC（Center for High Performance Computing）上准备环境、提交 Slurm 作业，并运行 SeekUI 的 Stage 1、Stage 2、推理和评估。

> 本文以 Notchpeak 上的 SoC GPU 资源为主。分区、QoS、GPU 类型和账户权限可能调整，提交前请用 `sinfo`、`sacctmgr` 重新确认。
>
> 原始 SeekUI Stage 2 脚本面向 LUMI-G，不是 CHPC。本文中的 CHPC 单卡 Stage 2 配置属于实验性适配，不代表已经完整复现原始训练。

## 1. 先了解 CHPC 的基本使用方式

CHPC 使用 Slurm 调度计算资源：

- 登录节点：用于编辑文件、查看日志、提交作业和轻量数据管理。
- 计算节点：用于训练、推理、编译大型 CUDA 扩展和较重的数据处理。
- Slurm 批处理脚本：通过 `#SBATCH` 指定账户、分区、GPU、CPU、内存和时间。
- `$HOME`：适合保存代码和小型配置，不建议存放大型数据集、模型缓存和大量 checkpoint。
- `/scratch/general/vast/$USER`：适合存放数据集、HuggingFace 缓存和训练输出，但重要结果仍应另行备份。

不要在登录节点直接运行长时间训练或全量推理。

### 1.1 建议的目录约定

以下示例假设仓库位于 `$HOME/SeekUI`。如果实际路径不同，请相应修改：

```bash
export SEEKUI_ROOT="$HOME/SeekUI"
export SCRATCH_ROOT="/scratch/general/vast/$USER"
export SEEKUI_WORK="$SCRATCH_ROOT/SeekUI"
export DATA_ROOT="$SEEKUI_WORK/data"

mkdir -p "$SEEKUI_WORK" "$DATA_ROOT"
mkdir -p "$SEEKUI_ROOT/logs"
```

后续示例使用以下路径：

```text
代码：                  $HOME/SeekUI
数据：                  /scratch/general/vast/$USER/SeekUI/data
Stage 2 DatasetDict：   /scratch/general/vast/$USER/SeekUI/vsgui_train_seekui_qwen2_5_explanation
Stage 1 权重：          /scratch/general/vast/$USER/SeekUI_sft
Stage 2 输出：          /scratch/general/vast/$USER/SeekUI/outputs/
```

## 2. 查看自己的账户和可用资源

### 2.1 查看账户和 QoS

```bash
sacctmgr show assoc where user=$USER \
  format=Cluster,Account,Partition,QOS,DefaultQOS
```

本文主要使用以下账户和分区：

| 用途 | account | partition/QoS | 时间限制与注意事项 |
|---|---|---|---|
| SoC GPU 作业 | `soc-gpu-np` | `soc-gpu-np` | 最长 12 小时 |
| Owner GPU guest | `owner-gpu-guest` | `notchpeak-gpu-guest` | 最长 3 天，可能被 owner 作业抢占 |
| 普通 GPU 分区 | `notchpeak-gpu` | `notchpeak-gpu` | 资源类型以 `sinfo` 为准 |
| SoC 短作业 | `soc-gpu-np` | QoS `soc-gpushort-np` | 最长 1 小时，适合显存测试 |

时间限制以当前 Slurm 配置为准，可以用下面的命令检查：

```bash
scontrol show partition soc-gpu-np
scontrol show partition notchpeak-gpu-guest
sacctmgr show qos where name=soc-gpushort-np
```

### 2.2 查看 GPU 节点

```bash
sinfo -p soc-gpu-np -N -o "%N %G %t %C %m"
sinfo -p notchpeak-gpu-guest -N -o "%N %G %t %C %m"
```

节点显示为 `mixed` 并不代表一定还有空闲 GPU。继续检查具体节点：

```bash
scontrol show node <NODE_NAME>
```

重点查看：

```text
Gres=...
CfgTRES=...gres/gpu=...
AllocTRES=...gres/gpu=...
```

`CfgTRES` 是总 GPU 数，`AllocTRES` 是已分配 GPU 数。

### 2.3 常见 GPU 资源名

| Slurm GRES | 显存/说明 |
|---|---|
| `gpu:h100nvl:1` | H100 NVL，约 94GB，适合优先尝试大模型训练 |
| `gpu:a100_80gb_pcie:1` | A100 80GB PCIe，主要见于 guest 资源 |
| `gpu:a100:1` | A100，具体显存以分配后的 `nvidia-smi` 为准 |
| `gpu:a6000:1` | RTX A6000，48GB |
| `gpu:a800:1` | 当前 SoC 节点为 A800 40GB |
| `gpu:l40:1` | L40，48GB |

Kingspeak 上主要是 P100，Lonepeak 上主要是 1080Ti，不建议用于原始全参数 Stage 2。

### 2.4 常用作业命令

```bash
# 提交作业
sbatch job.slurm

# 查看自己的作业
squeue -u $USER

# 查看某个等待作业的原因
squeue -j <JOBID> -o "%.18i %.10T %.20R %b %l"

# 查看作业历史和最终状态
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList

# 取消作业
scancel <JOBID>

# 查看日志
tail -f <LOG_FILE>
```

## 3. 准备 Conda 环境

如果已经创建好 `seekui` 环境，只需在每个 Slurm 作业中激活，不要重复安装。

### 3.1 首次安装

```bash
module load cuda/11.8
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

conda create -n seekui python=3.10.12 -y
conda activate seekui

pip install vllm==0.7.1 datasets==4.0.0 math-verify==0.8.0 trl==0.15.0
pip install matplotlib==3.10.0 scipy==1.13.1 mergekit llm_blender
pip install transformers==4.55.0 deepspeed==0.16.2 scikit-learn wandb==0.21.0
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install transformers==4.51.1 decord==0.6.0 qwen_vl_utils==0.0.14
```

最终生效的 Transformers 版本应为 **4.51.1**。`flash-attn` 编译较耗时，应安排在合适的计算环境，不要长时间占用登录节点。

检查环境：

```bash
python -c "import torch, transformers, trl, deepspeed; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('trl:', trl.__version__); print('deepspeed:', deepspeed.__version__)"
```

### 3.2 每个 GPU 作业都设置缓存

建议将持久缓存放在 scratch，将 Triton 临时缓存放在计算节点本地磁盘：

```bash
export HF_HOME="$SCRATCH_ROOT/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/torch_cache"
export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
```

这样可避免在 `$HOME` 中产生大量模型缓存，也可减少 NFS 缓存造成的卡顿。

## 4. 准备数据

完整数据说明见 [`data/README.md`](../data/README.md)。

### 4.1 Stage 1 数据

需要准备：

```text
$DATA_ROOT/scanpath_train_explanation.json
$DATA_ROOT/target2text.json
$DATA_ROOT/vsgui10k-images/
```

然后修改：

```text
stage1/qwen-vl-finetune/qwenvl/data/__init__.py
```

将 `VSGUI_TEXT` 配置为实际路径，例如：

```python
VSGUI_TEXT = {
    "annotation_path": "/scratch/general/vast/<UNID>/SeekUI/data/scanpath_train_explanation.json",
    "data_path": "/scratch/general/vast/<UNID>/SeekUI/data",
    "target": "/scratch/general/vast/<UNID>/SeekUI/data/target2text.json",
}
```

如果使用的是本地生成的 `scanpath_train_think.json`，则让 `annotation_path` 指向该文件。

### 4.2 Stage 2 DatasetDict

Stage 2 入口使用：

```python
DatasetDict.load_from_disk(script_args.dataset_name)
```

因此 `--dataset_name` 必须指向通过 `save_to_disk()` 创建的本地目录，而不只是普通 JSON 文件。

可以从 HuggingFace 加载后保存一次：

```bash
conda activate seekui

python - <<'PY'
import os
from datasets import load_dataset

output = os.path.join(
    os.environ["SEEKUI_WORK"],
    "vsgui_train_seekui_qwen2_5_explanation",
)
dataset = load_dataset("sushizixin1/vsgui_train_seekui_qwen2_5_explanation")
dataset.save_to_disk(output)
print(dataset)
print("Saved to:", output)
PY
```

也可以按照 `data/README.md`，使用 `stage2/dataset/build_visgui_dataset.py` 从原始数据构建。

## 5. Stage 1：SFT 训练

如果已经有 Stage 1 权重，可以直接跳到 Stage 2。

### 5.1 原始单卡入口

```text
stage1/qwen-vl-finetune/run_eye_tracking.sh
```

关键配置为：

- Qwen2.5-VL-3B-Instruct；
- 冻结 vision tower，训练 LLM 和 multimodal projector；
- BF16 + DeepSpeed ZeRO-3；
- 每卡 batch size 2；
- gradient accumulation 4；
- 200 epochs。

提交前检查脚本中的：

```bash
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR="/path/to/huggingface/hub"
OUTPUT_DIR="/path/to/SeekUI_sft"
DATASETS="vsgui_text%100"
```

### 5.2 H100 NVL 单卡 Slurm 示例

将下面内容保存为自己的 `stage1_h100.slurm`：

```bash
#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=seekui-stage1
#SBATCH --output=stage1_%j.out
#SBATCH --error=stage1_%j.err

module load cuda/11.8
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate seekui

export SEEKUI_ROOT="$HOME/SeekUI"
export SCRATCH_ROOT="/scratch/general/vast/$USER"
export HF_HOME="$SCRATCH_ROOT/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/torch_cache"
export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
cd "$SEEKUI_ROOT"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
bash stage1/qwen-vl-finetune/run_eye_tracking.sh
```

提交：

```bash
cd "$HOME/SeekUI"
sbatch stage1_h100.slurm
```

如果通过 guest 队列申请 A100 80GB，可将资源行替换为：

```bash
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --gres=gpu:a100_80gb_pcie:1
```

> 显存提醒：本项目历史上使用原始 Stage 1 单卡配置时，RTX A6000 48GB 曾在第一个 optimizer step 发生 OOM。因此不要把“模型只有 3B”理解为 48GB 一定能训练。单卡优先选择 A100 80GB 或 H100 NVL 94GB。

## 6. 原始 Stage 2 配置与 CHPC 的区别

原始 `main` 分支的 `stage2/src/seekui/run_eye_tracking.sh` 面向 LUMI-G：

| 项目 | 原始配置 |
|---|---|
| 节点 | 8 个 LUMI-G 节点 |
| GPU | 每节点 8 个 AMD MI250X GCD，共 64 个训练 rank |
| 物理 GPU | 32 张 MI250X，每个 GCD 64GB HBM |
| 训练方式 | 全参数 GRPO + 独立 reference model |
| 并行 | DeepSpeed ZeRO-3 |
| Offload | 无 optimizer/parameter offload |
| 每卡 batch | 1 prompt |
| Gradient accumulation | 1 |
| Generations | 每个 prompt 4 条 |
| 全局 batch | 64 prompts，即 256 completions/optimizer step |
| Epochs | 30 |
| Prompt/completion | 1024/512 tokens |
| Gradient checkpointing | false |
| vLLM | false |
| Learning rate / beta | TRL 默认值 1e-6 / 0.04 |
| 保存 | 每 500 steps，且 `save_only_model=true` |

因此，不能将“原始每 GPU 64GB 能运行”直接推导为“CHPC 单张 80GB 能运行”：原始配置依赖 64 个 rank 的 ZeRO-3 分片。

也不要直接对原始脚本执行 `sbatch`：其中的 `standard-g`、`project_`、路径和节点数都不是 CHPC 配置。

## 7. Stage 2：CHPC 单卡实验性适配

适用前提：

- 已经有 Stage 1 权重；
- 希望先验证 GRPO 数据、reward、生成和反向传播；
- 接受单卡配置需要实测，且完整 30 epochs 可能非常慢。

推荐资源顺序：

1. `1×H100 NVL 94GB`；
2. `1×A100 80GB`；
3. 多卡 ZeRO-3；
4. 如果接受非全参数实验，再考虑 LoRA，而不是继续强行压缩全参数显存。

### 7.1 创建带 optimizer CPU offload 的 ZeRO 配置

先复制原始配置：

```bash
cd "$HOME/SeekUI/stage2/src/seekui"
cp local_scripts/zero3.json local_scripts/zero3_chpc_single.json
```

将新文件中的：

```json
"offload_optimizer": {
    "device": "none",
    "pin_memory": true
}
```

改为：

```json
"offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
}
```

初次测试保留：

```json
"offload_param": {
    "device": "none",
    "pin_memory": true
}
```

如果仍然 OOM，再考虑将 parameter 也 offload 到 CPU，但速度会进一步下降。启用 offload 后建议申请约 160–200GB CPU 内存，并检查节点实际可用内存。

### 7.2 单卡 H100 小规模显存测试

下面示例只运行 2 个 optimizer steps，用于检查：

- 数据和模型能否加载；
- 4 条 completion 能否生成；
- model/reference forward 是否正常；
- 第一个 optimizer step 是否 OOM；
- checkpoint 是否能成功写盘。

```bash
#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100nvl:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --job-name=seekui-grpo-smoke
#SBATCH --output=grpo_smoke_%j.out
#SBATCH --error=grpo_smoke_%j.err

module load cuda/11.8
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate seekui

export SEEKUI_ROOT="$HOME/SeekUI"
export SCRATCH_ROOT="/scratch/general/vast/$USER"
export SEEKUI_WORK="$SCRATCH_ROOT/SeekUI"
export HF_HOME="$SCRATCH_ROOT/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/torch_cache"
export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export DEBUG_MODE=true
export LOG_PATH="$SEEKUI_WORK/outputs/grpo_smoke_reward.txt"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"
mkdir -p "$SEEKUI_WORK/outputs"

cd "$SEEKUI_ROOT/stage2/src/seekui"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

srun python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=4623 \
  src/open_r1/grpo.py \
  --output_dir "$SEEKUI_WORK/outputs/SeekUI_grpo_smoke" \
  --model_name_or_path "$SCRATCH_ROOT/SeekUI_sft" \
  --dataset_name "$SEEKUI_WORK/vsgui_train_seekui_qwen2_5_explanation" \
  --deepspeed ./local_scripts/zero3_chpc_single.json \
  --max_prompt_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 2 \
  --logging_steps 1 \
  --learning_rate 1e-6 \
  --beta 0.04 \
  --optim adamw_torch \
  --lr_scheduler_type linear \
  --warmup_ratio 0 \
  --weight_decay 0 \
  --bf16 true \
  --report_to wandb \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --max_pixels 401408 \
  --run_name SeekUI_grpo_smoke \
  --save_steps 2 \
  --save_only_model false \
  --save_total_limit 1 \
  --num_generations 4 \
  --max_completion_length 512
```

如果使用 A100 80GB guest 资源，替换 account、partition 和 GRES：

```bash
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --gres=gpu:a100_80gb_pcie:1
```

如需使用 1 小时 short QoS，可在 SoC 分区作业中添加：

```bash
#SBATCH --qos=soc-gpushort-np
#SBATCH --time=01:00:00
```

但生成 4×512 tokens 和保存完整 checkpoint 可能耗时较长，1 小时不保证足够。

### 7.3 从测试配置转为正式训练

测试成功后：

1. 删除 `--max_steps 2`；
2. 添加 `--num_train_epochs 30`；
3. 使用新的 `--output_dir`，不要覆盖测试输出；
4. 根据实测 seconds/step 调整 `save_steps`；
5. 保留 `save_only_model=false`，以支持完整续训；
6. 设置合理的 `save_total_limit`，并检查 scratch 配额。

全局 batch 的选择：

```text
global prompt batch
= GPU 数 × per_device_train_batch_size × gradient_accumulation_steps
```

原始配置的全局 prompt batch 是 64。单卡若要严格保持这个有效 batch，需要：

```bash
--gradient_accumulation_steps 64
```

但这会导致大量串行生成，训练非常慢。若改成 4、8 等较小值，必须明确这已经改变了原始训练配置。

### 7.4 多卡替代方案

如果单卡 OOM 或速度不可接受，可使用 2 张或更多 GPU，让 ZeRO-3 恢复分片能力。例如申请 2 张 A6000：

```bash
#SBATCH --gres=gpu:a6000:2
```

同时将训练启动参数改为：

```bash
--nproc_per_node=2
```

若每卡 batch 为 1，并希望保持原始全局 prompt batch 64，则：

```bash
--gradient_accumulation_steps 32
```

A6000 的生成速度低于 A100/H100，多卡是否更快获得资源取决于当时队列。A800 40GB、A6000 48GB 和 L40 48GB 都不应在未测试前视为与 A100 80GB 等价。

## 8. Checkpoint 与 12 小时任务续训

原始 Stage 2 配置：

```bash
--save_steps 500
--save_only_model true
```

不适合直接用于可能达到 12 小时上限或被抢占的 CHPC 作业：

- 如果单卡 500 steps 需要超过 12 小时，作业结束时可能一个 checkpoint 都没有；
- `save_only_model=true` 只保存模型，不完整保存 optimizer、scheduler 等训练状态；
- 从纯模型权重重新开始不等于无损续训。

CHPC 上建议：

```bash
--save_steps <根据实测速度设置>
--save_only_model false
--save_total_limit 2
```

完整 checkpoint 明显大于纯模型权重，应提前检查存储空间。

当前原始入口中是：

```python
trainer.train()
```

需要恢复某个完整 checkpoint 时，应在 `grpo.py` 中显式传入路径，例如：

```python
trainer.train(
    resume_from_checkpoint="/scratch/general/vast/<UNID>/SeekUI/outputs/SeekUI_grpo/checkpoint-100"
)
```

全新训练时恢复为 `trainer.train()`。不要只修改命令行而假设入口一定会自动续训。

对于 guest/抢占作业，务必让 checkpoint 间隔小于作业可能运行的时间，并给最后一次写盘预留时间。

## 9. LoRA 方案的定位

如果目标是比较 reward、数据或训练策略，而不是严格复现原始全参数 Stage 2，可以考虑 LoRA：

- 单卡 batch size 1；
- 开启 gradient checkpointing；
- 冻结基础模型，只训练 adapter；
- 非 ZeRO-3 的 PEFT 路径可以通过禁用 adapter 计算 reference 输出，避免常驻第二份完整模型。

LoRA 会改变可训练参数和实验设置，需要单独确认 PEFT target modules、学习率和保存/合并方式。本文不把它与原始全参数配置混为同一个复现实验。

## 10. 推理与评估

### 10.1 推理

在已分配 GPU 的计算作业中运行：

```bash
cd "$SEEKUI_ROOT/inference"

python prediction_think.py \
  --model_path "$SEEKUI_WORK/outputs/SeekUI_grpo" \
  --scanpath_test "$DATA_ROOT/scanpath_test.json" \
  --target2text "$DATA_ROOT/target2text.json" \
  --image_root "$DATA_ROOT" \
  --output "$SEEKUI_WORK/outputs/test_predictions_SeekUI.json"
```

也可以直接使用发布权重：

```text
sushizixin1/SeekUI_sft
sushizixin1/SeekUI
```

如果目的只是推理或评估，不需要重新训练 Stage 2。

### 10.2 评估

```bash
cd "$SEEKUI_ROOT/evaluation"

python evaluation.py \
  --prediction_file "$SEEKUI_WORK/outputs/test_predictions_SeekUI.json"
```

按 UI 类型评估：

```bash
python evaluation_types.py \
  --prediction_file "$SEEKUI_WORK/outputs/test_predictions_SeekUI.json" \
  --img2type_file "$DATA_ROOT/img_to_type.json"
```

已有 `clusters.npy` 时，不需要重新运行 `preprocess_SScluster.py`。

## 11. 常见问题排查

### 11.1 第一个 optimizer step 才 CUDA OOM

前向和反向成功不代表 optimizer states 能放下。优先检查：

1. 是否使用 batch size 1；
2. 是否开启 `--gradient_checkpointing true`；
3. 是否启用 optimizer CPU offload；
4. 是否同时存在训练模型和 reference model；
5. 是否仍保留 4 条 completion 的计算图；
6. 是否需要改用多卡 ZeRO-3 或 LoRA。

可设置以下环境变量缓解部分显存碎片，但它不能解决根本性的显存不足：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 11.2 图片文件不存在

检查：

```bash
ls "$DATA_ROOT/vsgui10k-images" | head
```

确认 JSON 中的相对路径与 `--image_root`/Stage 1 `data_path` 拼接后真实存在。scratch 中的数据如果被清理，需要从备份重新解压，不要等到训练开始后才处理。

### 11.3 Stage 2 模型加载类型不正确

当前 trainer 通过模型路径字符串判断是否使用 Qwen2.5-VL 加载逻辑。路径中包含以下字符串更稳妥：

```text
SeekUI
Qwen2.5-VL
```

例如 `SeekUI_sft` 可以匹配。如果 checkpoint 放在完全不包含这些字符串的目录中，应检查 `grpo_trainer.py` 的模型选择分支。

### 11.4 `DatasetDict.load_from_disk` 报错

确认 `--dataset_name` 指向包含 `dataset_dict.json` 的本地目录，而不是：

- 原始 JSON 文件；
- 普通 HuggingFace 缓存目录；
- 只有图片的目录；
- 未完成的下载目录。

### 11.5 多进程端口冲突

修改 Torch distributed 的端口，例如：

```bash
--master_port=4624
```

并确保同一节点上的其他作业没有使用相同端口。

### 11.6 作业达到时间上限，没有 checkpoint

根据实测速度降低 `save_steps`，并使用完整 checkpoint。对于非常慢的单卡 GRPO，原始的 500 steps 可能过于稀疏。

### 11.7 WandB 或缓存导致作业等待

离线环境使用：

```bash
export WANDB_MODE=offline
```

模型缓存放 scratch，Triton 临时缓存放节点本地 `/tmp`，避免多个进程共同等待缓慢的 home/NFS 路径。

## 12. 推荐的实际工作流程

1. 用 `sacctmgr` 和 `sinfo` 确认账户、GPU 类型和时间限制。
2. 准备并检查 Conda 环境，不在每个作业中重复安装依赖。
3. 将数据、模型缓存和输出放到 scratch，并备份重要 checkpoint。
4. 已有 Stage 1 权重时，先运行 2–5 个 Stage 2 optimizer steps。
5. 记录 GPU 型号、峰值显存、seconds/step 和 checkpoint 写盘时间。
6. 再决定使用单卡全参数、多卡 ZeRO-3，还是 LoRA。
7. 正式训练前设置可恢复的 checkpoint，不要把 12 小时全部押在 `save_steps=500` 上。
8. 训练完成后，再用独立 GPU 作业运行推理和评估。

最重要的原则是：**先用短作业获得真实显存和速度数据，再申请长时间资源；不要把原始 LUMI 64-rank 配置直接等价为 CHPC 单卡配置。**
