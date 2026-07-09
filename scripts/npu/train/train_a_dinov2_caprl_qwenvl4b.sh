# ====================== 实验说明 ======================


SCRIPT_PATH=$(readlink -f "$0") # 获取训练脚本绝对路径
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")   # 获取脚本所在文件夹路径
REPO_ROOT=$(readlink -f "${SCRIPT_DIR}/../../..") # 向上三级获取项目根目录路径

cd "${REPO_ROOT}" # 进入项目根目录

# ====================== 输出日志 ======================
# 日志配置放这里
USER_DIR=$(readlink -f "${REPO_ROOT}/..")
LOG_DIR="${USER_DIR}/logs"
mkdir -p ${LOG_DIR}
RUN_TIME=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${RUN_TIME}.log"
exec > >(tee -a ${LOG_FILE}) 2>&1

# 选择模型配方
DATASET_PHASE=phase_b # 数据集阶段：phase_a是单patch推理；phase_b是状态更新推理
MAP_TASK=lane_intersection # 任务类型: lane or lane_intersection.
VISION_BACKBONE=dinov2 # Visual backbone selector used by the generic multi-vision launcher.
# Vision assets for this recipe. Scripts only download the towers declared below.
VISION_TOWER_NAME=facebook_dinov2-large # Single vision tower directory name under MODEL_OBS_PATH.
MM_VISION_TOWER_TYPE=dinov2 # Model-side vision tower type: dinov2, dinov3, multi_moe, or multi_concat.
INPUT_IMAGE_SIZE=518 # Image size fed to the vision encoder; DINOv3 recipes usually use 512.

# # ====================== DI输出根路径 ======================
CLUSTER_SAVE=${OUTPUT_URL} # DI训练平台输出结果的保存路径
OSB_SHARE_PATH="${CLUSTER_SAVE}"

echo "============================================================"
echo "Script path: ${SCRIPT_PATH}"
echo "Repo root: ${REPO_ROOT}"
echo "User dir: ${USER_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "System defined obs share path: ${OSB_SHARE_PATH}"
echo "Recipe: ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
echo "============================================================"

# ====================== OBS下载路径 ======================
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
OBS_CACHE=${OBS_CACHE:-${USER_DIR}} # 昇腾服务器根目录
MODEL_OBS_PATH=${MODEL_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/checkpoints} # 从OBS中加载LLM与vision encoder权重
QWEN_PATH=${QWEN_PATH:-${OBS_CACHE}/checkpoints/inputs/CapRL-Qwen3VL-4B} # 本地的Qwen-VL模型路径
DATASET_OBS_PATH=${DATASET_OBS_PATH:-obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jjh/data/data_lane_intersection_norm_sample_512_33w.zip} # 从OBS中加载数据集
DATASET_DIR_NAME=${DATASET_DIR_NAME:-data_lane_intersection_norm_sample_512_33w} # 数据集解压后的目录名

# ====================== 服务器解压路径 ======================
VISION_TOWER=${VISION_TOWER:-${OBS_CACHE}/checkpoints/inputs/${VISION_TOWER_NAME}} # 昇腾服务器vision encoder权重路径
DATASET_ZIP_PATH=${DATASET_ZIP_PATH:-${OBS_CACHE}/dataset/inputs/dataset_${RUN_ID}.zip} # 昇腾服务器数据集路径
DATASET_EXTRACT_ROOT=${DATASET_EXTRACT_ROOT:-${OBS_CACHE}/dataset/inputs/dataset_extract_${RUN_ID}} #昇腾服务器数据集解压后的路径
DATASET_PATH=${DATASET_PATH:-${DATASET_EXTRACT_ROOT}/${DATASET_DIR_NAME}}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATASET_PATH}}

# ====================== DI输出路径 ======================
CLOUD_OUTPUT_PATH=${OSB_SHARE_PATH%/}/${RUN_ID} # DI平台输出在OBS的保存路径

# ====================== 昇腾输出路径 ======================
LOCAL_MODEL_SAVE_ROOT=${LOCAL_MODEL_SAVE_ROOT:-/cache/xyk/checkpoints/outputs} # 输出在昇腾服务器的根路径
LOCAL_MODEL_SAVE_PATH=${LOCAL_MODEL_SAVE_PATH:-${LOCAL_MODEL_SAVE_ROOT}/${RUN_ID}}  # 输出在昇腾服务器的保存路径

# ====================== training params ======================
TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE:-128}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-8}
LR=${LR:-2e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-2e-5}
MM_VISION_TOWER_LR=${MM_VISION_TOWER_LR:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-500}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/deepspeed_zero3.json}
ENABLE_EVAL=${ENABLE_EVAL:-False}
SAVE_BEST_EVAL_LOSS=${SAVE_BEST_EVAL_LOSS:-False}
SAVE_BEST_TRAIN_LOSS=${SAVE_BEST_TRAIN_LOSS:-True}
BEST_TRAIN_LOSS_START_STEP=${BEST_TRAIN_LOSS_START_STEP:-5000}                    
SAVE_BEST_INFER_INDEX=${SAVE_BEST_INFER_INDEX:-False}                             
BEST_INFER_INDEX_METRIC=${BEST_INFER_INDEX_METRIC:-length_f1}                     
BEST_INFER_INDEX_NUM_SAMPLES=${BEST_INFER_INDEX_NUM_SAMPLES:-0}                   
BEST_CHECKPOINT_SAVE_MODE=${BEST_CHECKPOINT_SAVE_MODE:-rotating_create_only}      
BEST_CHECKPOINT_KEEP_LIMIT=${BEST_CHECKPOINT_KEEP_LIMIT:-8}                       
VISION_LAYER_FUSION_INDEXES=${VISION_LAYER_FUSION_INDEXES:-}                      
VISION_LAYER_FUSION_TYPE=${VISION_LAYER_FUSION_TYPE:-mean}                       
SWANLAB_ENABLE=${SWANLAB_ENABLE:-True}                                            
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"5gIH7zqSwmo8dl1Ia5vRN"}                
SWANLAB_PROJECT=${SWANLAB_PROJECT:-unimapgen_v9}                                  
SWANLAB_GROUP=${SWANLAB_GROUP:-sft_phase_a_lane_intersection_dinov2_nodeepstack}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-sft_phase_a_lane_intersection_dinov2_qwen3vlcaprl4b_256_ep8}  
SWANLAB_TAGS=${SWANLAB_TAGS:-sft,phase_a,lane_intersection,dinov2,qwen3vl8b,nodeepstack,unimapgen_v9}  
SWANLAB_MODE=${SWANLAB_MODE:-offline}                                             
SWANLAB_API_HOST=${SWANLAB_API_HOST:-}                                            
SWANLAB_WEB_HOST=${SWANLAB_WEB_HOST:-}

# ====================== Ascend environment ======================
# Ascend and HCCL runtime environment for NPU jobs.
export ASCEND_CUSTOM_PATH=${ASCEND_CUSTOM_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend toolkit root.
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest}  # Ascend custom operator package root.
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}  # Ascend operator package path.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}                             # Network interface used by Gloo rendezvous.
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}                                 # Network interface used by tensor-parallel services.
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}                             # Network interface used by HCCL communication.
export CUDA_DEVICE_MAX_CONNECTIONS=1                                              # NPU compatibility setting used by Ascend PyTorch jobs.
export HCCL_WHITELIST_DISABLE=1                                                   # Disable HCCL whitelist checks on managed clusters.
export HCCL_CONNECT_TIMEOUT=7200                                                  # HCCL connection timeout in seconds.
export HCCL_EXEC_TIMEOUT=7200                                                     # HCCL execution timeout in seconds.
export HCCL_IF_BASE_PORT=64000                                                    # Base port for HCCL communication.
export INF_NAN_MODE_ENABLE=1                                                      # Enable Inf/NaN handling in Ascend runtime.
export HCCL_ASYNC_ERROR_HANDLING=0                                                # HCCL async error handling switch.
export WITHOUT_JIT_COMPILE=1                                                      # Disable JIT compile path for more stable NPU startup.
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE                                        # Disable HCCL FFTS operator base mode for compatibility.
export COMBINED_ENABLE=1                                                          # Ascend combined-operator switch used by the NPU runtime.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}                                      # CPU thread count per process.
export MLLM_LOG_RANK0_ONLY=${MLLM_LOG_RANK0_ONLY:-1}                              # Limit project logs to rank 0 when set.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}                    # Disable tokenizer worker parallelism warnings.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"                                  # Ensure project modules are importable.

# ====================== Dependency install ======================
INSTALL_DEPS=${INSTALL_DEPS:-False} # 是否在运行该脚本时安装依赖库
ENABLE_MOXING_UPGRADE=${ENABLE_MOXING_UPGRADE:-False} # 是否替换DI平台镜像环境中的moxing库
VLLM_VERSION=${VLLM_VERSION:-0.9.2} # vLLM version used by GRPO rollout workers.
VLLM_ASCEND_VERSION=${VLLM_ASCEND_VERSION:-0.9.2rc1} # vLLM-Ascend version used by GRPO rollout workers.

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
  pip install 'loguru>=0.7.0' 'shapely>=2.0.0' wandb "swanlab==0.7.19" "huggingface-hub==0.36.2" urllib3==1.26.15
fi

# ====================== 分布式环境初始化 ======================
# 分布式集群拓扑：本地默认配置 或 ModelArts 平台提供的节点元数据
if [[ -z "${MA_VJ_NAME:-}" ]]; then
  NNODES=${NNODES:-1} # 分布式训练总节点数量
  NODE_RANK=${NODE_RANK:-0} # 当前节点在分布式任务中的序号
  NPROC_PER_NODE=${NPROC_PER_NODE:-8} # 单节点上启动的NPU训练进程数
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} # 分布式通信主节点地址（进程组汇聚地址）
else
  NNODES=${NNODES:-$MA_NUM_HOSTS} # 分布式训练总节点数量，优先读取平台主机总数环境变量
  NODE_RANK=${NODE_RANK:-$VC_TASK_INDEX} # 当前节点在分布式任务中的序号，取自平台任务索引
  NPROC_PER_NODE=${NPROC_PER_NODE:-$MA_NUM_GPUS} # 单节点上启动的NPU训练进程数，取自平台单节点卡数
  MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}} # 分布式通信主节点地址，截取集群节点列表第一个IP作为主节点
fi
MASTER_PORT=${MASTER_PORT:-6060} # 分布式通信主节点端口
export NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export RDZV_ID=${RDZV_ID:-sft_phase_a_lane_intersection_dinov2_${RUN_ID}} # 当前分布式任务唯一进程组标识，避免多任务通信串扰

mkdir -p "${LOCAL_MODEL_SAVE_PATH}"
OUTPUT_PATH="${LOCAL_MODEL_SAVE_PATH}" # 传给训练脚本的实际输出根目录

echo "============================================================"
echo "Run id: ${RUN_ID}"
echo "Local output path: ${OUTPUT_PATH}"
echo "Cloud output path: ${CLOUD_OUTPUT_PATH}"
echo "============================================================"

SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${OUTPUT_PATH}/swanlab} # SwanLab可视化工具本地日志存放目录

# ====================== 资源下载环节 ======================
# 下载当前任务所需的模型权重与数据集文件，并校验本地必要路径是否存在
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/${VISION_TOWER_NAME}', '${VISION_TOWER}')"
python -c "import moxing as mox; mox.file.copy('${DATASET_OBS_PATH}', '${DATASET_ZIP_PATH}')"
mkdir -p "${DATASET_EXTRACT_ROOT}"
unzip -q "${DATASET_ZIP_PATH}" -d "${DATASET_EXTRACT_ROOT}"
python -c "import moxing as mox; mox.file.copy_parallel('${MODEL_OBS_PATH}/CapRL-Qwen3VL-4B', '${QWEN_PATH}')"
INIT_MODEL_PATH="${QWEN_PATH}" # 传入训练脚本train_qwen的初始模型路径，第二阶段训练会复用第一阶段产出的检查点
TRAIN_PATH="${DATASET_PATH}/${DATASET_PHASE}/train.jsonl" # 当前数据集阶段对应的训练集JSONL文件路径
EVAL_PATH="${DATASET_PATH}/${DATASET_PHASE}/eval.jsonl" # 当前数据集阶段对应的验证集JSONL文件路径
TEST_PATH="${DATASET_PATH}/${DATASET_PHASE}/test.jsonl" # 用于离线推理的测试集JSONL文件（历史兼容字段）
# 前置校验：若模型、数据集、图像、视觉主干等任意必备资源缺失，直接终止任务
for path in "${INIT_MODEL_PATH}" "${VISION_TOWER}" "${TRAIN_PATH}" "${EVAL_PATH}" "${IMAGE_FOLDER}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path not found: ${path}"
    exit 1
  fi
done

# 根据目标全局批次大小，自动计算梯度累积步数
TOTAL_DEVICES=$(( NNODES * NPROC_PER_NODE )) # 所有节点合计的NPU进程总数
MICRO_BATCH=$(( TOTAL_DEVICES * PER_DEVICE_TRAIN_BATCH_SIZE )) # 梯度累积前，单次迭代实际全局微批次大小
GRADIENT_ACCUMULATION_STEPS=$(( (TARGET_GLOBAL_BATCH_SIZE + MICRO_BATCH - 1) / MICRO_BATCH )) # 计算所需梯度累积步数，凑出目标全局批次大小
if [ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]; then
  GRADIENT_ACCUMULATION_STEPS=1 # 梯度累积步数最小强制设为1
fi

# 组装可选评测、视觉层融合参数数组，用于传给Python训练入口脚本
# 根据当前transformers版本自动适配评测策略参数名，新旧版本兼容
EVAL_STRATEGY_ARG=$(python -c "import inspect, transformers; print('--eval_strategy' if 'eval_strategy' in inspect.signature(transformers.TrainingArguments.__init__).parameters else '--evaluation_strategy')")
EVAL_ARGS=()                                                                      # 训练器可选评测参数，仅开启评测时才填充
if [[ "${ENABLE_EVAL}" =~ ^(1|true|True|TRUE|yes|YES)$ ]]; then
  EVAL_ARGS=(                                                                     # 开启评测后填入的全套评测相关命令行参数
    --eval_data_path "${EVAL_PATH}"
    --eval_image_folder "${IMAGE_FOLDER}"
    "${EVAL_STRATEGY_ARG}" steps
    --eval_steps "${EVAL_STEPS}"
    --save_best_eval_loss "${SAVE_BEST_EVAL_LOSS}"
    --best_eval_loss_dir eval_best
  )
fi

VISION_LAYER_FUSION_ARGS=() # 视觉编码器层融合可选参数数组
if [ -n "${VISION_LAYER_FUSION_INDEXES}" ]; then
  VISION_LAYER_FUSION_ARGS=(
    # 配置了融合层序号时，追加视觉层融合参数
    --vision_layer_fusion_indexes ${VISION_LAYER_FUSION_INDEXES}
    --vision_layer_fusion_type "${VISION_LAYER_FUSION_TYPE}"
  )
fi

# 在启动耗时较长的训练任务前，打印解析完成后的全部运行配置信息
echo "============================================================"
echo "任务配置:       ${DATASET_PHASE} | ${MAP_TASK} | ${VISION_BACKBONE}"
echo "初始大模型路径:   ${INIT_MODEL_PATH}"
echo "视觉编码器路径: ${VISION_TOWER}"
echo "视觉层融合: ${VISION_LAYER_FUSION_INDEXES:-关闭} (融合类型:${VISION_LAYER_FUSION_TYPE})"
echo "训练集路径:        ${TRAIN_PATH}"
echo "验证集路径:         ${EVAL_PATH}"
echo "输出保存目录:       ${OUTPUT_PATH}"
echo "============================================================"

# 启动训练入口脚本，分布式训练后端使用HCCL/DDP，完整SFT支持DeepSpeed优化
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m mllm.train.train_qwen \
  --model_name_or_path "${INIT_MODEL_PATH}" \
  --version conv_qwen_3_Dinov2_huawei \
  --vision_tower "${VISION_TOWER}" \
  --mm_vision_tower_type "${MM_VISION_TOWER_TYPE}" \
  --input_image_size "${INPUT_IMAGE_SIZE}" \
  "${VISION_LAYER_FUSION_ARGS[@]}" \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --unfreeze_mm_vision_tower True \
  --disable_deepstack True \
  --data_path "${TRAIN_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  "${EVAL_ARGS[@]}" \
  --sample_seed 42 \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_PATH}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --mm_projector_lr "${MM_PROJECTOR_LR}" \
  --mm_vision_tower_lr "${MM_VISION_TOWER_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --remove_unused_columns false \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --save_best_train_loss "${SAVE_BEST_TRAIN_LOSS}" \
  --best_train_loss_start_step "${BEST_TRAIN_LOSS_START_STEP}" \
  --best_train_loss_dir best \
  --save_best_infer_index "${SAVE_BEST_INFER_INDEX}" \
  --best_infer_index_dir infer_best \
  --best_infer_index_metric "${BEST_INFER_INDEX_METRIC}" \
  --best_infer_index_phase "${DATASET_PHASE}" \
  --best_infer_index_eval_data_path "${EVAL_PATH}" \
  --best_infer_index_image_folder "${IMAGE_FOLDER}" \
  --best_infer_index_vision_tower "${VISION_TOWER}" \
  --best_infer_index_input_image_size "${INPUT_IMAGE_SIZE}" \
  --best_infer_index_conv_template conv_qwen_3_Dinov2_huawei \
  --best_infer_index_map_task "${MAP_TASK}" \
  --best_infer_index_num_samples "${BEST_INFER_INDEX_NUM_SAMPLES}" \
  --best_infer_index_eval_steps "${SAVE_STEPS}" \
  --best_infer_index_max_new_tokens 2048 \
  --best_checkpoint_save_mode "${BEST_CHECKPOINT_SAVE_MODE}" \
  --best_checkpoint_keep_limit "${BEST_CHECKPOINT_KEEP_LIMIT}" \
  --use_hf_progress_bar True \
  --logging_steps "${LOGGING_STEPS}" \
  --report_to none \
  --swanlab_enable "${SWANLAB_ENABLE}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_group "${SWANLAB_GROUP}" \
  --swanlab_job_type sft \
  --swanlab_tags "${SWANLAB_TAGS}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --swanlab_log_dir "${SWANLAB_LOG_DIR}" \
  --swanlab_api_host "${SWANLAB_API_HOST}" \
  --swanlab_web_host "${SWANLAB_WEB_HOST}" \
  --ddp_find_unused_parameters False \
  --ddp_backend hccl \
  --deepspeed "${DEEPSPEED_CONFIG}"


# 捕获训练进程退出码，后续执行清理、上传逻辑前先保存该状态码
TRAIN_EXIT=$?
# 判断训练是否异常退出
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "训练任务异常终止，退出码：${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

# 仅主节点（rank0）执行本地训练产物迁移至云端存储路径
if [[ "${NODE_RANK}" == "0" ]]; then
  # 校验云端目标目录是否已存在，防止覆盖原有数据
  if [ -e "${CLOUD_OUTPUT_PATH}" ]; then
    echo "错误：云端输出路径已存在，禁止覆盖：${CLOUD_OUTPUT_PATH}"
    exit 1
  fi
  echo "开始将主节点本地训练产物迁移至云端：${OUTPUT_PATH} -> ${CLOUD_OUTPUT_PATH}"
  mv "${OUTPUT_PATH}" "${CLOUD_OUTPUT_PATH}"
  MOVE_EXIT=$?
  # 校验迁移操作是否执行成功
  if [ "${MOVE_EXIT}" -ne 0 ]; then
    echo "错误：本地目录迁移至云端失败，操作退出码：${MOVE_EXIT}"
    exit "${MOVE_EXIT}"
  fi
  echo "训练产物已成功迁移至云端路径：${CLOUD_OUTPUT_PATH}"
else
  # 非主节点无需执行上传迁移，直接跳过
  echo "当前节点编号 ${NODE_RANK} 为从节点，跳过云端产物迁移步骤"
fi