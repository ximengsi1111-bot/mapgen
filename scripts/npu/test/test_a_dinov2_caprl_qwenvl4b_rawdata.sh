#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# 验证脚本（原始数据版）
# phase_a | lane_intersection | dinov2 + CapRL-Qwen3VL-4B
# 接收已切好的 lane_ins_png/ 和 test/*.jsonl
# 移除 OBS 下载、eval、可视化等依赖
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")
cd "${REPO_ROOT}"

# ====================== 日志 ======================
USER_DIR="${REPO_ROOT}/.."
LOG_DIR="${USER_DIR}/logs"
mkdir -p ${LOG_DIR}
RUN_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_rawdata_${RUN_TIME}.log"
exec > >(tee -a ${LOG_FILE}) 2>&1

# ====================== 配置标识 ======================
DATASET_PHASE=phase_a
MAP_TASK=lane_intersection
VISION_BACKBONE=dinov2
VISION_TOWER_NAME=facebook_dinov2-large
MM_VISION_TOWER_TYPE=dinov2
INPUT_IMAGE_SIZE=518

echo "脚本路径: ${SCRIPT_PATH}"
echo "项目根目录: ${REPO_ROOT}"
echo "配置: ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"

# ====================== 数据集路径 ======================
# 挂载原始数据集目录（内部应包含 sample 子目录，每个 sample 下有
# infer_patch_tif/、lane_ins_res/、lane_ins_png/、test/）
DATASET_ROOT=${DATASET_ROOT:-${REPO_ROOT}/dataset}
echo "数据集根目录: ${DATASET_ROOT}"

# ====================== OBS / 模型路径 ======================
CLUSTER_SAVE=${OUTPUT_URL:-}
OBS_CACHE=${OBS_CACHE:-${USER_DIR}}
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/inputs/${VISION_TOWER_NAME}}
CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-}
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-}
# 推理结果输出
# 推理结果直接写入各样本的 lane_ins_res/{big_stem}/ 目录

# ====================== 推理参数 ======================
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
COORD_MODE=${COORD_MODE:-auto}
COORD_RANGE=${COORD_RANGE:-1000}

# ====================== Ascend 环境 ======================
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export INF_NAN_MODE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export WITHOUT_JIT_COMPILE=1
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE
export COMBINED_ENABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ====================== 分布式设置 ======================
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}
  NODE_RANK=${NODE_RANK:-0}
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
fi
MASTER_PORT=${MASTER_PORT:-6060}
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-test_phase_a_lane_intersection_caprl4b_rawdata_${RUN_TIME}}

# ====================== Vision Tower ======================
mkdir -p "$(dirname "${VISION_TOWER}")"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"

# ====================== 辅助函数 ======================
read_list() {
  python - "$1" <<'PY'
import re
import sys

for item in re.split("[,;" + chr(10) + "]+", sys.argv[1] or ""):
    item = item.strip()
    if item:
        print(item)
PY
}

safe_label() {
  python - "$1" <<'PY'
import re
import sys

value = sys.argv[1].strip().rstrip("/") or "checkpoint"
label = re.sub(r"[^A-Za-z0-9._-]+", "_", value.split("/")[-1]).strip("._-")
print(label or "checkpoint")
PY
}

# ====================== 解析检查点列表 ======================
CHECKPOINT_ITEMS=()
CHECKPOINT_LABELS=()
if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
  while IFS= read -r obs_item; do
    label=$(safe_label "${obs_item}")
    local_dir="${OBS_CACHE}/checkpoints/inputs/checkpoint_${label}_${RUN_ID:-${RUN_TIME}}"
    python -c "import moxing as mox; mox.file.copy_parallel('${obs_item}', '${local_dir}')"
    CHECKPOINT_INPUT_PATH="${local_dir}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors")):
    print(root)
    raise SystemExit(0)
cmd = [
    sys.executable,
    "scripts/tools/resolve_best_checkpoint.py",
    "--output-dir", str(root),
    "--best-name", "infer_best",
    "--best-name", "eval_best",
    "--best-name", "best",
    "--best-name", "best_reward",
    "--allow-direct",
]
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
if result.returncode == 0 and result.stdout.strip():
    print(result.stdout.strip())
    raise SystemExit(0)
checkpoints = []
for path in root.glob("checkpoint-*"):
    if path.is_dir():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        checkpoints.append((step, path))
if checkpoints:
    print(sorted(checkpoints)[-1][1])
    raise SystemExit(0)
raise SystemExit(f"cannot resolve checkpoint under: {root}")
PY
)
    CHECKPOINT_ITEMS+=("${RESOLVED_CHECKPOINT}")
    CHECKPOINT_LABELS+=("${label}")
  done < <(read_list "${CHECKPOINT_OBS_LIST}")
elif [ -n "${CHECKPOINT_DIRS}" ]; then
  while IFS= read -r local_item; do
    CHECKPOINT_INPUT_PATH="${local_item}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"checkpoint path does not exist: {root}")
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors")):
    print(root)
    raise SystemExit(0)
cmd = [
    sys.executable,
    "scripts/tools/resolve_best_checkpoint.py",
    "--output-dir", str(root),
    "--best-name", "infer_best",
    "--best-name", "eval_best",
    "--best-name", "best",
    "--best-name", "best_reward",
    "--allow-direct",
]
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
if result.returncode == 0 and result.stdout.strip():
    print(result.stdout.strip())
    raise SystemExit(0)
checkpoints = []
for path in root.glob("checkpoint-*"):
    if path.is_dir():
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except Exception:
            step = -1
        checkpoints.append((step, path))
if checkpoints:
    print(sorted(checkpoints)[-1][1])
    raise SystemExit(0)
raise SystemExit(f"cannot resolve checkpoint under: {root}")
PY
)
    CHECKPOINT_ITEMS+=("${RESOLVED_CHECKPOINT}")
    CHECKPOINT_LABELS+=("$(safe_label "${local_item}")")
  done < <(read_list "${CHECKPOINT_DIRS}")
else
    echo "Error: set CHECKPOINT_OBS_LIST or CHECKPOINT_DIRS"
    exit 1
fi

# ====================== 验证 Vision Tower ======================
if [ ! -e "${VISION_TOWER}" ]; then
    echo "Error: vision tower not found: ${VISION_TOWER}"
    exit 1
fi

# ====================== 单检查点推理 ======================
# 对数据集中的每个样本的每张大图执行推理
# test/*.jsonl 由 split_raw_patches.py 预生成
run_one_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_label="$2"

  # 收集 dataset 下所有样本的 test/*.jsonl
  TEST_JSON_FILES=()
  while IFS= read -r -d '' f; do
    TEST_JSON_FILES+=("$f")
  done < <(find "${DATASET_ROOT}" -mindepth 3 -maxdepth 5 -name "*.jsonl" -path "*/test/*" -print0 2>/dev/null)

  if [ ${#TEST_JSON_FILES[@]} -eq 0 ]; then
    echo "Error: no test/*.jsonl found under ${DATASET_ROOT}"
    echo "  Run split_raw_patches.py first."
    exit 1
  fi

  for test_json in "${TEST_JSON_FILES[@]}"; do
    sample_dir=$(dirname "$(dirname "${test_json}")")
    sample_id=$(basename "${sample_dir}")
    big_stem=$(basename "${test_json%.jsonl}")
    echo "推理 ${checkpoint_label}: sample=${sample_id}, image=${big_stem}"
    echo "  jsonl: ${test_json}"

    local results_dir="${sample_dir}/lane_ins_res/${big_stem}"
    local json_dir="${results_dir}/json"
    mkdir -p "${results_dir}" "${json_dir}"

    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --node_rank="${NODE_RANK}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        scripts/tools/infer_centerline_checkpoint.py \
        --checkpoint-dir "${checkpoint_dir}" \
        --vision_tower "${VISION_TOWER}" \
        --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
        --input_image_size "${INPUT_IMAGE_SIZE}" \
        --disable_deepstack \
        --test-json "${test_json}" \
        --num-samples "${NUM_TEST_SAMPLES}" \
        --image-folder "${DATASET_ROOT}" \
        --prompt-mode dataset \
        --map-task "${MAP_TASK}" \
        --patch-size 256 \
        --coord-mode "${COORD_MODE}" \
        --coord-range "${COORD_RANGE}" \
        --conv-template conv_qwen_3_Dinov2_huawei \
        --output-dir "${results_dir}" \
        --sample-json-dir "${json_dir}" \
        --output-json "${results_dir}/summary.json" \
        --temperature 0.0 \
        --max-new-tokens "${MAX_NEW_TOKENS}"
  done
}



# ====================== 执行 ======================
echo "运行 ID: ${RUN_ID:-${RUN_TIME}}"

for index in "${!CHECKPOINT_ITEMS[@]}"; do
  label="${CHECKPOINT_LABELS[$index]}"
  checkpoint="${CHECKPOINT_ITEMS[$index]}"
  run_one_checkpoint "${checkpoint}" "${label}"
done

echo "推理完成"
echo "结果已写入各样本的 lane_ins_res/ 目录下"
