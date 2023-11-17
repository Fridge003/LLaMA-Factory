MODEL_DIR="./model"
DATA_DIR="alpaca_en"

ZERO_STAGE=2
NUM_GPUS=4
BS_PER_GPU=16
#GRAD_ACCUM=$((512/$BS_PER_GPU))
GRAD_ACCUM=1
LOG_STEP=1
MAX_STEP=2

mkdir -p ./logs
LOG_DIR="./logs/zero_${ZERO_STAGE}_numgpu_${NUM_GPUS}_bs_${BS_PER_GPU}"
mkdir -p $LOG_DIR
OUTPUT_LOG="${LOG_DIR}/output.log"
TFLOPS_LOG="${LOG_DIR}/tflops_profiler.log"
MEMORY_LOG="${LOG_DIR}/memory_profiler.log"
DEEPSPEED_CONFIG="${LOG_DIR}/zero_${ZERO_STAGE}_numgpu_${NUM_GPUS}_bs_${BS_PER_GPU}.json"

cat <<EOT > $DEEPSPEED_CONFIG
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "flops_profiler": {
        "enabled": true,
        "profile_step": $LOG_STEP,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": false,
        "output_file": "$TFLOPS_LOG"
    }
}
EOT
# "fp16": {
#     "enabled": "auto",
#     "loss_scale": 0,
#     "initial_scale_power": 16,
#     "loss_scale_window": 1000,
#     "hysteresis": 2,
#     "min_loss_scale": 1
# }, 

deepspeed --num_gpus $NUM_GPUS --master_port=9901 src/train_bash.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --stage pt \
    --model_name_or_path $MODEL_DIR \
    --do_train \
    --dataset $DATA_DIR \
    --template default \
    --finetuning_type full \
    --output_dir ./ckpt \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size $BS_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --bf16 True \
    --learning_rate 5e-5 \
    --max_steps $MAX_STEP \
    --save_steps 1000 \
    --flash_attn True \
    --plot_loss \
    --memory_profiling_step $LOG_STEP \
    --memory_profiling_log $MEMORY_LOG 2>&1 | tee $OUTPUT_LOG