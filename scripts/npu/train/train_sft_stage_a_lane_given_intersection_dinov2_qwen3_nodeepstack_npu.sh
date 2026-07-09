#!/usr/bin/env bash
set -euo pipefail

# phase_a | lane_given_intersection | DINOv2 | Qwen3 | no DeepStack

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

: "${OUTPUT_URL:?OUTPUT_URL is required on the training platform}"

DATASET_PHASE=phase_a
MAP_TASK=lane_given_intersection
VISION_RECIPE=dinov2
MODEL_LABEL=qwen3

CLUSTER_SAVE=${OUTPUT_URL}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-/cache}

VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/facebook_dinov2-large}
MM_VISION_TOWER_TYPE=dinov2
INPUT_IMAGE_SIZE=${INPUT_IMAGE_SIZE:-518}
QWEN_PATH=${QWEN_PATH:-${OBS_CACHE}/checkpoints/Qwen3-8B}

TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-6}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-500}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
LORA_ENABLE=${LORA_ENABLE:-False}
DISABLE_DEEPSTACK=${DISABLE_DEEPSTACK:-True}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}; NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}; MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi
MASTER_PORT=${MASTER_PORT:-6060}
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
OUTPUT_PATH=${LOCAL_MODEL_SAVE_PATH:-/tmp/output_${RUN_ID}}
mkdir -p "${OUTPUT_PATH}"

TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE ))
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE ))
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH ))
[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ] && GRADIENT_ACCUMULATION_STEPS=1

echo "Launching lane_given_intersection phase=${DATASET_PHASE} model=${QWEN_PATH}"

torchrun \
  --nnodes="${NNODES}" --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  --model_name_or_path "${QWEN_PATH}" \
  --map_task lane_given_intersection \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
  --input_image_size "${INPUT_IMAGE_SIZE}" \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower True \
  --disable_deepstack "${DISABLE_DEEPSTACK}" \
  --data_path "${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" \
  --image_folder "${IMAGE_FOLDER}" \
  --sample_seed 42 --image_aspect_ratio pad --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --lora_enable "${LORA_ENABLE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True --dataloader_num_workers 4 \
  --remove_unused_columns false \
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit 15 \
  --use_hf_progress_bar True --logging_steps 10 --report_to none \
  --ddp_find_unused_parameters False --ddp_backend hccl \
  --deepspeed "${DEEPSPEED_CONFIG}"
