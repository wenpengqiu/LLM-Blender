import os
import sys
import logging
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset

# 设置日志
logger = logging.getLogger(__name__)

# ==========================================
# 1. 答案提取逻辑
# ==========================================

def extract_mmlu_answer(text):
    if not text: return None
    # 尝试分割 Final Answer
    parts = re.split(r'Final Answer[:\s]*', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        target_text = parts[-1].strip()
        match = re.search(r'^[\(\[]?([A-D])[\)\]\.]?', target_text)
        if match: return match.group(1)
    # 查找 (A) 或 A. 格式
    matches = re.findall(r'[\(\s]([A-D])[\)\.]', text)
    if matches: return matches[-1]
    # 查找独立的 A-D
    matches = re.findall(r'\b([A-D])\b', text)
    if matches: return matches[-1]
    return None

def extract_gsm8k_answer(text):
    if not text: return None
    def parse_num(num_str):
        try:
            clean_str = num_str.replace(',', '')
            if clean_str.endswith('.'): clean_str = clean_str[:-1]
            if '.' in clean_str: return float(clean_str)
            return int(clean_str)
        except: return None
    
    # 优先查找 #### 后的答案
    if "####" in text:
        parts = text.split("####")
        match = re.search(r'(-?[\d,]+(?:\.\d+)?)', parts[-1])
        if match: return parse_num(match.group(1))
    
    # 查找 Final Answer
    parts = re.split(r'Final Answer[:\s]', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        target_text = parts[-1]
        matches = re.findall(r'-?[\d,]+(?:\.\d+)?', target_text)
        if matches: return parse_num(matches[-1])
        
    # 最后的兜底：查找文中最后一个数字
    matches = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if matches: return parse_num(matches[-1])
    return None

# ==========================================
# 2. 参数定义
# ==========================================

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/flan-t5-xl",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )

@dataclass
class DataArguments:
    train_file: str = field(
        default=None, metadata={"help": "The input training data file (a jsonl file)."}
    )
    validation_file: str = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a jsonl file)."}
    )
    max_source_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."}
    )
    max_target_length: int = field(
        default=512,
        metadata={"help": "The maximum total sequence length for target text after tokenization."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    # 新增参数：指定任务类型以选择正确的提取逻辑
    task_name: str = field(
        default="gsm8k",
        metadata={"help": "Task name for metrics computation: 'gsm8k' or 'mmlu'"}
    )

def main():
    # 1. 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 设置随机种子
    set_seed(training_args.seed)

    # 3. 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # 4. 加载数据集
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    
    raw_datasets = load_dataset("json", data_files=data_files)

    # 5. 加载模型配置和分词器
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # 6. 数据预处理
    def preprocess_function(examples):
        inputs = examples["input"]
        targets = examples["output"]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=data_args.max_source_length, 
            padding="max_length", 
            truncation=True
        )
        
        labels = tokenizer(
            text_target=targets, 
            max_length=data_args.max_target_length, 
            padding="max_length", 
            truncation=True
        )

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["validation"].column_names,
                desc="Running tokenizer on validation dataset",
            )

    # 7. Metrics 计算函数
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # 1. 确保 preds 是 numpy 数组 (有时候可能是 tensor)
        if not isinstance(preds, np.ndarray):
            preds = preds.detach().cpu().numpy()
            
        # 2. 清洗 preds：将 -100 替换为 pad_token_id
        # 防止 OverflowError
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        
        # 解码生成结果
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # 解码标签 (将 -100 替换回 pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 打印前 3 条生成的预测 vs 真实标签
        logger.info("\n" + "="*50)
        logger.info("DEBUG: Generated Predictions Sample:")
        for i in range(min(3, len(decoded_preds))):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Pred: {decoded_preds[i]}")
            logger.info(f"  Gold: {decoded_labels[i]}")
            logger.info("-" * 20)
        logger.info("="*50 + "\n")
        
        score = 0
        total = len(decoded_preds)
        
        for pred_text, label_text in zip(decoded_preds, decoded_labels):
            is_correct = False
            task = data_args.task_name.lower()

            if task == "gsm8k":
                pred_ans = extract_gsm8k_answer(pred_text)
                label_ans = extract_gsm8k_answer(label_text)
                # 数值比较，容差 1e-6
                if pred_ans is not None and label_ans is not None and abs(pred_ans - label_ans) < 1e-6:
                    is_correct = True
            
            elif task == "mmlu":
                pred_ans = extract_mmlu_answer(pred_text)
                label_ans = extract_mmlu_answer(label_text)
                # 选项比较 (忽略大小写)
                if pred_ans and label_ans and str(pred_ans).upper() == str(label_ans).upper():
                    is_correct = True
            
            else:
                # 默认 Exact Match
                if pred_text.strip() == label_text.strip():
                    is_correct = True
            
            if is_correct:
                score += 1
                
        return {"accuracy": score / total if total > 0 else 0}

    # 8. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        ),
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # 9. 训练与评估
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
