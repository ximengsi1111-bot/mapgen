#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# NPU 推理评估
# 固定配方：phase_a | 车道+路口 | dinov2 + CapRL-Qwen3VL-4B | 无 DeepStack
# 本文件自包含，不调用其他项目 .sh 文件。
# 对应训练脚本：scripts/npu/train/train_sft_stage_a_lane_intersection_dinov2_qwen3vl_caprl4b_nodeepstack_npu.sh
# ============================================================

SCRIPT_PATH=$(readlink -f "$0")                                                   # 本启动脚本的绝对路径。
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")                                              # 本启动脚本所在的目录。
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..")                                 # 项目根目录，用于相对脚本和 Python 导入。
# 平台 I/O 和配方元数据优先声明，便于云上作业审计。
cd "${REPO_ROOT}"

# ====================== 输出日志 ======================
# 日志配置放这里
USER_DIR="${REPO_ROOT}/.."
LOG_DIR="${USER_DIR}/logs"
mkdir -p ${LOG_DIR}
RUN_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_${RUN_TIME}.log"
exec > >(tee -a ${LOG_FILE}) 2>&1

# 配方标识：固定任务、视觉架构、模型系列和训练变体。
DATASET_PHASE=phase_a                                                             # 数据集阶段：phase_a 为 patch 推理，phase_b 为状态更新。
MAP_TASK=lane_intersection                                                        # 任务类型：lane 或 lane_intersection。
VISION_BACKBONE=dinov2                                                            # 视觉骨干选择器，由通用多视觉启动器使用。
# 本配方的视觉资产。脚本仅下载下面声明的视觉塔。
VISION_TOWER_NAME=facebook_dinov2-large                                           # MODEL_OBS_PATH 下的视觉塔目录名。
MM_VISION_TOWER_TYPE=dinov2                                                       # 模型侧视觉塔类型：dinov2、dinov3、multi_moe 或 multi_concat。
INPUT_IMAGE_SIZE=518                                                              # 送入视觉编码器的图像尺寸；DINOv2 要求为 14 的整数倍。

echo "脚本路径: ${SCRIPT_PATH}"
echo "项目根目录: ${REPO_ROOT}"
echo "配方: ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
# ====================== 云存储路径 ======================
# OUTPUT_URL 由云训练平台注入。
# 保持参考脚本惯例：将 OUTPUT_URL 镜像为 OSB_SHARE_PATH。
# 云和本地存储根。输出先暂存到本地，再上传至 OBS。
CLUSTER_SAVE=${OUTPUT_URL}                                                        # 云训练平台注入的云输出根目录。
OSB_SHARE_PATH="${CLUSTER_SAVE}"                                                  # 现有脚本使用的平台输出根目录别名。
echo "系统定义的 OBS 共享路径: ${OSB_SHARE_PATH}"

# 推理先写本地文件，然后 rank0 上传完整结果目录。
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)} # 本次运行的唯一 ID，用于本地缓存和云输出目录。
OBS_CACHE=${OBS_CACHE:-${USER_DIR}} # 本地工作节点缓存根，存放模型、数据集、检查点和输出。
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints}  # 存储模型和视觉检查点资产的 OBS 目录。
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_lane_intersection_samples_norm_33w_empty_patch.zip}  # 已制备的 UniMapGen 数据集 OBS zip 路径。
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_lane_intersection_samples_norm_33w_empty_patch}                       # zip 解压后期望的数据集目录名。

CHECKPOINT_OBS_LIST=${CHECKPOINT_OBS_LIST:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/03/e6bd1c3d440b489a874ab05f78f4271a/output/20260703_082540/best_candidates/best_step-00021110_loss-0p1011} # 逗号、分号或换行分隔的待评估 OBS 检查点根路径。
CHECKPOINT_DIRS=${CHECKPOINT_DIRS:-} # 逗号、分号或换行分隔的待评估本地检查点根路径。
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/inputs/${VISION_TOWER_NAME}} # 传给模型加载器的视觉塔路径。多视觉时使用逗号列表。
# 本次运行的本地数据集和输出路径。
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset/inputs/dataset_${RUN_ID}.zip} # 下载的数据集 zip 本地路径。
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset/inputs/dataset_extract_${RUN_ID}}  # 数据集 zip 解压的本地目录。
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}} # 解压后的数据集根目录，包含 phase_a 和 phase_b 子文件夹。
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}} # 传给推理的图像根目录，通常即 DATASET_PATH。
TEST_JSON=${TEST_JSON:-${DATASET_PATH}/${DATASET_PHASE}/test.jsonl} # 所选数据集阶段的推理 JSONL 路径。
CHECKPOINT_DOWNLOAD_ROOT=${CHECKPOINT_DOWNLOAD_ROOT:-${OBS_CACHE}/checkpoints/inputs/checkpoints_${RUN_ID}}  # 从 OBS 下载检查点候选的本地根目录。
LOCAL_OUTPUT_ROOT=${LOCAL_OUTPUT_ROOT:-${OBS_CACHE}/results/test_phase_a_lane_intersection_dinov2_caprl4b_output_${RUN_ID}}  # 每次运行的本地推理输出根目录。
CLOUD_OUTPUT_DIR=${TEST_RESULT_OBS:-${OSB_SHARE_PATH%/}/test_results_${RUN_ID}}   # 最终的云输出目录，用于推理或 GRPO 结果。

# ====================== 推理参数 ======================
# CHECKPOINT_OBS_LIST 或 CHECKPOINT_DIRS 可包含一个或多个检查点。
# NUM_TEST_SAMPLES=0 表示运行完整测试 jsonl。
# Patch json、可视化、指标和拼接地图先写本地，再上传。
# 主要运行时参数和超参数。
NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES:-0}                                           # 待运行的测试样本数；0 表示完整测试集。
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}                                            # 每个样本生成的最大 token 数。
COORD_MODE=${COORD_MODE:-auto}                                                    # 坐标模式：auto 读取 meta.coord_mode，或强制 norm1000 或 pixel。
COORD_RANGE=${COORD_RANGE:-1000}                                                  # 归一化标签的坐标范围，通常为 1000。
# ====================== Ascend 环境 ======================
# Ascend 和 HCCL 运行时环境，用于 NPU 作业。
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend 工具包根目录。
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend 自定义算子包根目录。
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend 算子包路径。
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}                             # Gloo 会合使用的网络接口。
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}                                 # 张量并行服务使用的网络接口。
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}                             # HCCL 通信使用的网络接口。
export CUDA_DEVICE_MAX_CONNECTIONS=1                                              # Ascend PyTorch 作业的 NPU 兼容性设置。
export HCCL_WHITELIST_DISABLE=1                                                   # 禁用托管集群上的 HCCL 白名单检查。
export HCCL_CONNECT_TIMEOUT=7200                                                  # HCCL 连接超时（秒）。
export HCCL_EXEC_TIMEOUT=7200                                                     # HCCL 执行超时（秒）。
export HCCL_IF_BASE_PORT=64000                                                    # HCCL 通信的起始端口。
export INF_NAN_MODE_ENABLE=1                                                      # 启用 Ascend 运行时的 Inf/NaN 处理。
export HCCL_ASYNC_ERROR_HANDLING=0                                                # HCCL 异步错误处理开关。
export WITHOUT_JIT_COMPILE=1                                                      # 禁用 JIT 编译路径，提高 NPU 启动稳定性。
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE                                        # 禁用 HCCL FFTS 算子基础模式，确保兼容性。
export COMBINED_ENABLE=1                                                          # NPU 运行时使用的 Ascend 组合算子开关。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}                                      # 每个进程的 CPU 线程数。
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}                              # 设置后仅 rank 0 输出项目日志。
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # 禁用 tokenizer 工作线程并行警告。
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # 确保项目模块可导入。

# 托管 NPU 镜像的依赖安装。预构建镜像请设置 INSTALL_DEPS=False。
INSTALL_DEPS=${INSTALL_DEPS:-False}                                                # 是否在启动前安装 Python 依赖。
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False}                              # 是否用固定版本替换平台 moxing。
VLLM_VERSION=${VLLM_VERSION:-0.9.2}                                               # GRPO rollout 工作器使用的 vLLM 版本。
VLLM_ASCEND_VERSION=${VLLM_ASCEND_VERSION:-0.9.2rc1}                              # GRPO rollout 工作器使用的 vLLM-Ascend 版本。

if [[ "${ENABLE_MOXING_UPGRADE}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  USE_MEMARTS=0 python -c "import moxing; moxing.file.copy('obs://yw-ads-training-gy1/data/external/personal/00592907/dataset_index/pkgs/moxing_framework-2.3.8-py2.py3-none-any.250714.whl', '/home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl')"
  pip uninstall moxing-framework -y
  pip cache purge
  pip install /home/ma-user/moxing_framework-2.3.8-py2.py3-none-any.whl
  export MOX_PROFILE=1
  export MOX_RECORD_OBS=1
fi

if [[ "${INSTALL_DEPS}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  pip install torch==2.7.1 torch_npu==2.7.1rc1
  python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-ads-training-gy1/data/external/personal/w00886412/llm4drive_utils/torch_npu/whl/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl', '/home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl')"
  pip install --force-reinstall /home/ma-user/torch_npu-2.7.1.dev20250724-cp311-cp311-manylinux_2_28_aarch64.whl
  pip install "sentencepiece>=0.1.99" "tiktoken>=0.7.0" "transformers==4.56.2" "tokenizers>=0.22.0,<0.23.0"
  pip install accelerate==1.6.0 deepspeed==0.14.4 "safetensors>=0.4.3" packaging "Pillow>=10.0.0" torchvision==0.22.1
  pip install shortuuid "peft>=0.10.0" pydantic 'markdown2[all]' 'numpy>=1.26' 'scipy>=1.10' 'scikit-learn>=1.2'
  pip install requests uvicorn fastapi 'einops>=0.6' 'einops-exts>=0.0.4' 'timm>=0.9.0' 'opencv-python-headless>=4.8.0'
    pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb swanlab "huggingface-hub==0.36.2" urllib3==1.26.15

fi
# 辅助函数：解析逗号/分号/换行分隔的列表，以及生成安全的检查点标签。
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
# ====================== 分布式设置 ======================
# 分布式拓扑：本地默认值或 ModelArts 提供的节点元数据。
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1}                                                             # 分布式节点数。
  NODE_RANK=${NODE_RANK:-0}                                                       # 本节点在分布式作业中的 rank。
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}                                             # 每个节点上的 NPU 工作进程数。
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}                                           # 分布式会合主地址。
else
  NNODES=${NNODES:-$MA_NUM_HOSTS}                                                 # 分布式节点数。
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX}                                          # 本节点在分布式作业中的 rank。
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS}                                  # 每个节点上的 NPU 工作进程数。
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}                              # 分布式会合主地址。
fi
MASTER_PORT=${MASTER_PORT:-6060}                                                  # 分布式会合主端口。
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-test_phase_a_lane_intersection_caprl4b_${RUN_ID}}       # 本次分布式运行的唯一会合 ID。
# 下载视觉编码器和数据集到本地缓存，然后验证所需的本地路径。
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}" "${CHECKPOINT_DOWNLOAD_ROOT}" "${LOCAL_OUTPUT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
echo "运行 ID: ${RUN_ID}"
echo "本地输出根目录: ${LOCAL_OUTPUT_ROOT}"
echo "云输出目录: ${CLOUD_OUTPUT_DIR}"

# 从 OBS 根路径或本地目录构建检查点评估列表。
# 解析优先级：直接权重文件 > infer_best > eval_best > best > best_reward > 最新 step checkpoint
CHECKPOINT_ITEMS=()                                                               # 已解析的待评估检查点路径。
CHECKPOINT_LABELS=()                                                              # 与 CHECKPOINT_ITEMS 配对的显示标签。
if [ -n "${CHECKPOINT_OBS_LIST}" ]; then
  while IFS= read -r obs_item; do
    label=$(safe_label "${obs_item}")
    local_dir="${CHECKPOINT_DOWNLOAD_ROOT}/${label}"
    python -c "import moxing as mox; mox.file.copy_parallel('${obs_item}', '${local_dir}')"
    CHECKPOINT_INPUT_PATH="${local_dir}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'                  # 通过解析器选择检查点目录。
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"检查点路径不存在: {root}")
# 如果目录中直接包含模型权重文件，则无需进一步解析。
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors")):
    print(root)
    raise SystemExit(0)
# 调用解析脚本，按优先级查找最优检查点：infer_best > eval_best > best > best_reward。
cmd = [
    sys.executable,
    "scripts/tools/resolve_best_checkpoint.py",
    "--output-dir",
    str(root),
    "--best-name",
    "infer_best",
    "--best-name",
    "eval_best",
    "--best-name",
    "best",
    "--best-name",
    "best_reward",
    "--allow-direct",
]
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
if result.returncode == 0 and result.stdout.strip():
    print(result.stdout.strip())
    raise SystemExit(0)
# 兜底：选择 step 编号最大的 checkpoint-* 子目录。
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
raise SystemExit(f"无法在以下路径下解析到可用的检查点: {root}")
PY
)
    CHECKPOINT_ITEMS+=("${RESOLVED_CHECKPOINT}")
    CHECKPOINT_LABELS+=("${label}")
    done < <(read_list "${CHECKPOINT_OBS_LIST}")
elif [ -n "${CHECKPOINT_DIRS}" ]; then
  while IFS= read -r local_item; do
    CHECKPOINT_INPUT_PATH="${local_item}"
RESOLVED_CHECKPOINT=$(python - "${CHECKPOINT_INPUT_PATH}" <<'PY'                  # 通过解析器选择检查点目录。
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
if not root.exists():
    raise SystemExit(f"检查点路径不存在: {root}")
# 如果目录中直接包含模型权重文件，则无需进一步解析。
if any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors")):
    print(root)
    raise SystemExit(0)
# 调用解析脚本，按优先级查找最优检查点：infer_best > eval_best > best > best_reward。
cmd = [
    sys.executable,
    "scripts/tools/resolve_best_checkpoint.py",
    "--output-dir",
    str(root),
    "--best-name",
    "infer_best",
    "--best-name",
    "eval_best",
    "--best-name",
    "best",
    "--best-name",
    "best_reward",
    "--allow-direct",
]
result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
if result.returncode == 0 and result.stdout.strip():
    print(result.stdout.strip())
    raise SystemExit(0)
# 兜底：选择 step 编号最大的 checkpoint-* 子目录。
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
raise SystemExit(f"无法在以下路径下解析到可用的检查点: {root}")
PY
)
    CHECKPOINT_ITEMS+=("${RESOLVED_CHECKPOINT}")
    CHECKPOINT_LABELS+=("$(safe_label "${local_item}")")
  done < <(read_list "${CHECKPOINT_DIRS}")
else
    echo "错误：请设置 CHECKPOINT_OBS_LIST 或 CHECKPOINT_DIRS 以指定推理检查点。"
    exit 1
fi

# 验证推理所需的视觉塔、测试数据和图像文件夹路径是否存在。
for path in "${VISION_TOWER}" "${TEST_JSON}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "错误：找不到所需路径：${path}"
    exit 1
  fi
done

# 对单个检查点执行推理、评估、可视化和指标表打印。
# 参数：
#   $1 - checkpoint_dir: 检查点目录路径
#   $2 - checkpoint_label: 检查点显示标签（用于日志输出）
#   $3 - output_dir: 推理结果输出目录
# 输出目录结构：
#   json/              - 逐样本推理 JSON
#   viz/               - Patch 级可视化
#   whole_map_viz/     - 全局地图可视化
#   summary.json       - 推理摘要
#   merged_global.json - 合并全局结果
#   eval.json          - 评估指标
run_one_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_label="$2"
  local output_dir="$3"
  local json_dir="${output_dir}/json"
  local patch_viz_dir="${output_dir}/viz"
  local whole_map_viz_dir="${output_dir}/whole_map_viz"
  local summary_json="${output_dir}/summary.json"
  local merged_global_json="${output_dir}/merged_global.json"
  local eval_json="${output_dir}/eval.json"
  mkdir -p "${json_dir}" "${patch_viz_dir}" "${whole_map_viz_dir}"
    echo "推理 ${checkpoint_label}: ${checkpoint_dir}"
# 使用 torchrun 启动分布式推理。
# 注意：数据集图像为 256×256，BitImageProcessor 会自动 resize 到 INPUT_IMAGE_SIZE=518 送入视觉编码器；
#       --patch-size 256 仅用于坐标反归一化（与训练数据的归一化基准一致），不影响图像输入尺寸。
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
    --test-json "${TEST_JSON}" \
    --num-samples "${NUM_TEST_SAMPLES}" \
    --image-folder "${IMAGE_FOLDER}" \
    --prompt-mode dataset \
    --map-task "${MAP_TASK}" \
    --patch-size 256 \
    --coord-mode "${COORD_MODE}" \
    --coord-range "${COORD_RANGE}" \
    --conv-template conv_qwen_3_Dinov2_huawei \
    --output-dir "${output_dir}" \
    --sample-json-dir "${json_dir}" \
    --output-json "${summary_json}" \
    --temperature 0.0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --eval-centerline \
    --eval-output-json "${eval_json}"
  # 仅 rank0 执行后处理（可视化与评估指标打印）。
  if [ "${NODE_RANK}" -ne 0 ]; then
    return 0
  fi
  # Patch 级可视化 + 全局地图拼接可视化。
  python scripts/tools/visualize_centerline.py \
      --input-dir "${output_dir}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-dir "${patch_viz_dir}" \
      --eval-output-json "${eval_json}" \
      --whole-map-viz-dir "${whole_map_viz_dir}" \
      --map-task lane_intersection
  # 打印车道+路口评估指标表格。
  if [ -f "${eval_json}" ]; then
    python - "${eval_json}" <<'PY'
import json
import sys
from pathlib import Path
from infer_index.line_eval import print_lane_intersection_eval_tables
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
map_eval = payload.get('map_eval', payload) if isinstance(payload, dict) else payload
print_lane_intersection_eval_tables(map_eval)
PY
  fi
}

# 评估每个请求的检查点，多个检查点时分离输出目录。
for index in "${!CHECKPOINT_ITEMS[@]}"; do
  label="${CHECKPOINT_LABELS[$index]}"
  checkpoint="${CHECKPOINT_ITEMS[$index]}"
  if [ "${#CHECKPOINT_ITEMS[@]}" -gt 1 ]; then
    output_dir="${LOCAL_OUTPUT_ROOT}/${index}_${label}"
  else
    output_dir="${LOCAL_OUTPUT_ROOT}"
  fi
  run_one_checkpoint "${checkpoint}" "${label}" "${output_dir}"
done

# Rank 0 将完整的本地结果树上传至 OBS。
if [ "${NODE_RANK}" -eq 0 ]; then
  python -c "import moxing as mox; mox.file.copy_parallel('${LOCAL_OUTPUT_ROOT}', '${CLOUD_OUTPUT_DIR}')"
  echo "推理结果已上传至 ${CLOUD_OUTPUT_DIR}"
fi