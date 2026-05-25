EXPERIMENT_NAME="Selector_loss_edit_ver"
EXP_DIR="./experiments/${EXPERIMENT_NAME}"
mkdir -p ${EXP_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=. python scripts/train_rm.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B" \
    --k_dimensions 8 \
    --lambda_neg 1.0 \
    --beta_dir 0.2 \
    --target_tau 0.8 \
    --num_pos_heads 5 \
    --train_data_path "data/ultrafeedback_train.jsonl" \
    --eval_data_path "data/ultrafeedback_val.jsonl" \
    --output_dir ${EXP_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_steps 240 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --max_length 1024 \
    --bf16 True \
    --run_name ${EXPERIMENT_NAME}
