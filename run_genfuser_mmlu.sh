#!/bin/bash

# === 1. 路径配置 (MMLU) ===
DATA_DIR="./data/gen_fuser_data/mmlu"
OUTPUT_DIR="./checkpoints/genfuser_mmlu"
CACHE_DIR="./hf_models"
MODEL_NAME="google/flan-t5-xl" 

# === 2. 启动训练 ===
python train_genfuser.py \
    --model_name_or_path $MODEL_NAME \
    --train_file "$DATA_DIR/train.jsonl" \
    --validation_file "$DATA_DIR/val.jsonl" \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --task_name "mmlu" \
    --learning_rate 5e-5 \
    --optim adafactor \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --max_source_length 2560 \
    --max_target_length 1024 \
    --do_train \
    --do_eval \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_steps 50 \
    --overwrite_output_dir \
    --bf16 True \
    --save_total_limit 1 \
    --predict_with_generate True \
    --generation_max_length 1024 \
    --load_best_model_at_end True \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --report_to "none"