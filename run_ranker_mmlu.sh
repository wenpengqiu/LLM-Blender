#!/bin/bash

# 1. 定义数据路径
DATA_DIR="./data/pair_ranker_data/mmlu"
# 2. 定义输出路径
OUTPUT_DIR="./checkpoints/ranker_mmlu"
# 3. 缓存路径
CACHE_DIR="/data2/qwp/LLM-Blender/hf_models"
# 4. 基础模型配置
MODEL_NAME="microsoft/deberta-v3-large" 

# 5. 启动训练
python train_ranker.py \
    --ranker_type pairranker \
    --model_type deberta \
    --model_name $MODEL_NAME \
    --train_data_path "$DATA_DIR/train.jsonl" \
    --eval_data_path "$DATA_DIR/val.jsonl" \
    --test_data_path "$DATA_DIR/val.jsonl" \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --candidate_maxlength 1280 \
    --source_maxlength 1280 \
    --n_candidates 6 \
    --using_metrics "accuracy" \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_steps 50 \
    --overwrite_output_dir True \
    --fp16 True \
    --load_best_model_at_end True \
    --metric_for_best_model "dev_score" \
    --save_total_limit 1