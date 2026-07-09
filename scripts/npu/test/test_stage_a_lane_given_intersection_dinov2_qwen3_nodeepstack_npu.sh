#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NPU inference
# Fixed recipe: phase_a | lane_given_intersection | DINOv2 | Qwen3 | no DeepStack
# ============================================================

SCRIPT_PATH=
SCRIPT_DIR=
REPO_ROOT=
cd "{REPO_ROOT}"

: "{OUTPUT_URL:?OUTPUT_URL is required on the training platform}"

DATASET_phase_a=phase_a
MAP_TASK=lane_given_intersection
VISION_RECIPE=dinov2
MODEL_LABEL=qwen3

CLUSTER_SAVE={OUTPUT_URL}
RUN_ID={RUN_ID:-(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE={OBS_CACHE:-/cache}

VISION_TOWER={VISION_TOWER:-{OBS_CACHE}/checkpoints/facebook_dinov2-large}
MM_VISION_TOWER_TYPE=dinov2
INPUT_IMAGE_SIZE={INPUT_IMAGE_SIZE:-518}

DATASET_PATH={DATASET_PATH:-{OBS_CACHE}/data}
IMAGE_FOLDER={IMAGE_FOLDER:-{DATASET_PATH}}
TEST_JSON={TEST_JSON:-{DATASET_PATH}/phase_a/test.jsonl}
LOCAL_OUTPUT_ROOT={LOCAL_OUTPUT_ROOT:-{OBS_CACHE}/test_phase_a_lane_given_intersection}
CLOUD_OUTPUT_DIR={CLOUD_OUTPUT_DIR:-{OBS_CACHE}/test_results_phase_a_lane_given_intersection}
MAX_NEW_TOKENS={MAX_NEW_TOKENS:-2048}
DISABLE_DEEPSTACK={DISABLE_DEEPSTACK:-True}

export PYTHONPATH="{REPO_ROOT}:{PYTHONPATH:-}"

if [[ -z "{MA_VJ_NAME:-}" ]]; then
  NNODES={NNODES:-1}; NODE_RANK={NODE_RANK:-0}; NPROC_PER_NODE={NPROC_PER_NODE:-8}
  MASTER_ADDR={MASTER_ADDR:-127.0.0.1}
else
  NNODES={NNODES:-MA_NUM_HOSTS}; NODE_RANK={NODE_RANK:-VC_TASK_INDEX}
  NPROC_PER_NODE={NPROC_PER_NODE:-MA_NUM_GPUS}; MASTER_ADDR={MASTER_ADDR:-{VC_WORKER_HOSTS%%,*}}
fi
MASTER_PORT={MASTER_PORT:-6060}
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
mkdir -p "{LOCAL_OUTPUT_ROOT}"

VISION_ARGS=(--vision_tower "{VISION_TOWER}" --mm_vision_tower_type "{MM_VISION_TOWER_TYPE}" --input_image_size "{INPUT_IMAGE_SIZE}")
[[ "{DISABLE_DEEPSTACK}" =~ ^(1|true|True|TRUE|yes|YES)$ ]] && VISION_ARGS+=(--disable_deepstack)

echo "[test] phase_a | lane_given_intersection | {VISION_RECIPE} | {MODEL_LABEL}"

torchrun \
  --nnodes="{NNODES}" --nproc_per_node="{NPROC_PER_NODE}" \
  --node_rank="{NODE_RANK}" --master_addr="{MASTER_ADDR}" --master_port="{MASTER_PORT}" \
  scripts/tools/infer_centerline_checkpoint.py \
  --checkpoint-dir "{CHECKPOINT_DIR:-{1?checkpoint_dir required}}" \
  {VISION_ARGS[@]} \
  --test-json "{TEST_JSON}" \
  --image-folder "{IMAGE_FOLDER}" \
  --prompt-mode dataset \
  --map_task "{MAP_TASK}" \
  --conv-template lane_given_intersection \
  --patch-size 256 --coord-mode auto --coord-range 1000 \
  --output-dir "{LOCAL_OUTPUT_ROOT}" \
  --sample-json-dir "{LOCAL_OUTPUT_ROOT}/json" \
  --output-json "{LOCAL_OUTPUT_ROOT}/summary.json" \
  --temperature 0.0 --max-new-tokens "{MAX_NEW_TOKENS}" \
  --eval-centerline --eval-output-json "{LOCAL_OUTPUT_ROOT}/eval.json"

if [ "{NODE_RANK}" -eq 0 ] && [ -n "{CLOUD_OUTPUT_DIR:-}" ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('{LOCAL_OUTPUT_ROOT}', '{CLOUD_OUTPUT_DIR}')"
  echo "Results uploaded to {CLOUD_OUTPUT_DIR}"
fi
